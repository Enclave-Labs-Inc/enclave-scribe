#!/usr/bin/env bash
# Runs automatically on the EC2 instance via UserData.
# Installs deps, clones the repo, preps data, and starts training in tmux.
set -euo pipefail

HOME_DIR="/home/ubuntu"
REPO_DIR="$HOME_DIR/enclave-scribe"
LOG="$HOME_DIR/setup.log"

exec > >(tee -a "$LOG") 2>&1
echo "=== EnclaveScribe Instance Setup ==="
date

# ── Wait for cloud-init's unattended-upgrades to release the dpkg lock ────────
echo "Waiting for dpkg lock to be released..."
while fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1 \
   || fuser /var/lib/apt/lists/lock >/dev/null 2>&1 \
   || fuser /var/lib/dpkg/lock >/dev/null 2>&1; do
    sleep 10
done
echo "Lock released, proceeding."

# ── System deps ───────────────────────────────────────────────────────────────
apt-get update -qq
apt-get install -y -qq git tmux htop libreoffice-common libreoffice-writer \
    libreoffice-impress python3-pip > /dev/null

# ── Repo ─────────────────────────────────────────────────────────────────────
if [ ! -d "$REPO_DIR" ]; then
    git clone https://github.com/Enclave-Labs-Inc/enclave-scribe.git "$REPO_DIR"
fi
cd "$REPO_DIR"

# ── Python env ───────────────────────────────────────────────────────────────
pip install --quiet uv
uv venv --python 3.12 .venv
source .venv/bin/activate

# Unsloth — detects CUDA version automatically
pip install --quiet "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --quiet trl

# Project deps
pip install --quiet -e ".[train,eval]"

# ── HuggingFace login (token injected via environment or manually) ─────────────
if [ -n "${HF_TOKEN:-}" ]; then
    huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential
fi

# ── Weights & Biases login ────────────────────────────────────────────────────
pip install --quiet wandb
if [ -n "${WANDB_API_KEY:-}" ]; then
    wandb login "$WANDB_API_KEY"
else
    echo "WARNING: WANDB_API_KEY not set — training will log to wandb offline only."
    wandb offline
fi

echo "=== Setup complete. Starting pipeline in tmux ==="
date

# ── Start training pipeline in tmux ─────────────────────────────────────────
tmux new-session -d -s training -x 220 -y 50

tmux send-keys -t training "cd $REPO_DIR && source .venv/bin/activate" Enter
tmux send-keys -t training "echo '=== Data prep started ===' && bash scripts/prepare/run_all.sh" Enter
tmux send-keys -t training "echo '=== Training started ===' && torchrun --nproc_per_node=4 scripts/train.py --config configs/train/unsloth_aws.yaml" Enter

echo "Pipeline running in tmux session 'training'."
echo "Connect with: ssh -i KEY.pem ubuntu@IP then: tmux attach -t training"

# ── S3 checkpoint sync (runs in background, every 10 min) ────────────────────
if [ -n "${S3_CHECKPOINT_BUCKET:-}" ]; then
    tmux new-window -t training -n sync
    tmux send-keys -t training:sync \
        "while true; do aws s3 sync $REPO_DIR/outputs s3://$S3_CHECKPOINT_BUCKET/outputs --quiet; sleep 600; done" \
        Enter
    echo "Checkpoint sync → s3://$S3_CHECKPOINT_BUCKET/outputs every 10 min"
fi
