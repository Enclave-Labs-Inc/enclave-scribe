#!/usr/bin/env bash
# Run this once after SSH-ing into a Lambda Labs 2x A100 80GB instance.
#
# Usage:
#   ssh ubuntu@<instance-ip>
#   bash <(curl -s https://raw.githubusercontent.com/Enclave-Labs-Inc/enclave-scribe/main/scripts/lambda/setup.sh)
#
# Or after cloning:
#   bash scripts/lambda/setup.sh
#
# What it does:
#   1. Installs system deps (LibreOffice for PPT/DOCX, tmux)
#   2. Clones the repo (if not already present)
#   3. Creates a venv, installs Unsloth + all project deps
#   4. Starts data prep + training in a tmux session named 'training'
#
# Instance: Lambda Labs 2x A100 80GB SXM4 — $3.98/hr
# Expected runtime: ~20 hrs total (~$80)

set -euo pipefail

REPO_URL="https://github.com/Enclave-Labs-Inc/enclave-scribe.git"
REPO_DIR="$HOME/enclave-scribe"
LOG="$HOME/setup.log"

exec > >(tee -a "$LOG") 2>&1
echo "=== EnclaveScribe Lambda Setup ==="
date

# ── System deps ──────────────────────────────────────────────────────────────
sudo apt-get update -qq
sudo apt-get install -y -qq tmux htop libreoffice-common libreoffice-writer \
    libreoffice-impress git > /dev/null
echo "System deps installed."

# ── Repo ─────────────────────────────────────────────────────────────────────
if [ ! -d "$REPO_DIR" ]; then
    git clone "$REPO_URL" "$REPO_DIR"
    echo "Repo cloned → $REPO_DIR"
else
    cd "$REPO_DIR" && git pull
    echo "Repo updated."
fi
cd "$REPO_DIR"

# ── Python env ───────────────────────────────────────────────────────────────
# Lambda instances ship with Python 3.10+ and CUDA 12.x pre-configured
pip install --quiet uv
uv venv --python 3.11 .venv
source .venv/bin/activate

echo "Installing Unsloth..."
pip install --quiet "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --quiet trl

echo "Installing project deps..."
pip install --quiet -e ".[train,eval]"

# ── Optional: HuggingFace login ───────────────────────────────────────────────
if [ -n "${HF_TOKEN:-}" ]; then
    huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential
    echo "HuggingFace login complete."
fi

echo ""
echo "=== Setup complete. Launching pipeline in tmux ==="
date

# ── Pipeline in tmux ─────────────────────────────────────────────────────────
tmux new-session -d -s training -x 220 -y 50

tmux send-keys -t training "cd $REPO_DIR && source .venv/bin/activate" Enter
tmux send-keys -t training "echo '=== Step 1: Data prep ===' && bash scripts/prepare/run_all.sh 2>&1 | tee log/data_prep.log" Enter
tmux send-keys -t training "echo '=== Step 2: Training ===' && torchrun --nproc_per_node=2 scripts/train.py --config configs/train/lora_lambda_2xa100.yaml 2>&1 | tee log/train.log" Enter
tmux send-keys -t training "echo '=== Step 3: Eval ===' && python scripts/eval.py --model outputs/enclave-scribe-v1 --config configs/infer/default.yaml 2>&1 | tee log/eval.log" Enter
tmux send-keys -t training "echo '=== ALL DONE ==='" Enter

echo ""
echo "Pipeline running. Attach with:"
echo "  tmux attach -t training"
echo ""
echo "Monitor GPU:"
echo "  watch -n5 nvidia-smi"
