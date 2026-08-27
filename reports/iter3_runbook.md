# Iter-3 runbook — Indic OCR fine-tune

End-to-end steps to run iter-3 on AWS. All commands are the copy-paste sequence.

**Objective**: LoRA fine-tune `allenai/olmOCR-2-7B-1025` on Indic OCR data to close the Devanagari gap that Phase 0 revealed.

**Budget expectation**: ~$50-80 total (prep + training + eval on g5.12xlarge spot).

---

## Precondition: rotate keys if not done since last session

- W&B API key at [wandb.ai](https://wandb.ai) → Settings → revoke, create new
- HuggingFace token at [hf.co/settings/tokens](https://hf.co/settings/tokens) → delete old, create new

---

## Step 1 — Launch training instance (~5 min)

**On your local machine.** Same size as iter-2 since Indic dataset is large (~58 GB from HF):

```bash
INSTANCE_ID=$(aws ec2 run-instances \
  --image-id ami-012ba162b9cd2729c \
  --instance-type g5.12xlarge \
  --key-name enclave-scribe-key \
  --security-group-ids $(aws ec2 describe-security-groups \
    --filters "Name=group-name,Values=enclave-scribe-sg" \
    --region us-east-1 --query "SecurityGroups[0].GroupId" --output text) \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":500,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
  --count 1 \
  --region us-east-1 \
  --query "Instances[0].InstanceId" \
  --output text)
echo "Instance: $INSTANCE_ID"
echo "$INSTANCE_ID" > /tmp/iter3_instance_id.txt

aws ec2 wait instance-running --instance-ids $INSTANCE_ID --region us-east-1
PUBLIC_IP=$(aws ec2 describe-instances --instance-ids $INSTANCE_ID \
  --region us-east-1 --query "Reservations[0].Instances[0].PublicIpAddress" --output text)
echo "IP: $PUBLIC_IP"
```

Then upload your reference gazette PDF (for later eval):
```bash
scp -i enclave-scribe-key.pem \
  '/Users/alias1623/Downloads/36_SO3977(E)_14082018.pdf' \
  ubuntu@$PUBLIC_IP:~/gazette.pdf
ssh -i enclave-scribe-key.pem ubuntu@$PUBLIC_IP
```

---

## Step 2 — Instance setup (~10 min)

```bash
cd ~
git clone https://github.com/Enclave-Labs-Inc/enclave-scribe.git
cd enclave-scribe

# Ubuntu default python is 3.10; upgrade to 3.11 (required by olmocr and
# newer transformers pin used by iter-3)
sudo apt update && sudo apt install -y python3.11 python3.11-venv python3.11-dev

python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install pymupdf transformers torch pillow peft accelerate datasets huggingface_hub tqdm pyyaml wandb liger-kernel

# Set rotated tokens
export HF_TOKEN=<NEW-rotated-hf-token>
export WANDB_API_KEY=<NEW-rotated-wandb-key>

# Verify
python -c "
import torch, transformers, datasets
print(f'torch {torch.__version__}, cuda={torch.cuda.is_available()}, gpus={torch.cuda.device_count()}')
print(f'transformers {transformers.__version__}')
print(f'datasets {datasets.__version__}')
"
# Expect: cuda=True, gpus=4
```

---

## Step 3 — Data prep (~1-2 hrs, ~50 GB disk)

Uses himalaya-ai dataset streaming. Saves images to disk + JSONL pairs. Small default sample cap keeps prep fast for a first run:

```bash
tmux new -s prep

# Inside tmux — re-export env vars (fresh shell)
export HF_TOKEN=<NEW-rotated-hf-token>
export WANDB_API_KEY=<NEW-rotated-wandb-key>

cd ~/enclave-scribe && source .venv/bin/activate

# Prep himalaya-ai Indic OCR data — 30k samples (adjust up if time budget allows)
python scripts/prepare/prep_himalaya_indic.py \
  --raw_dir data/raw \
  --out_jsonl data/interim/himalaya_indic.jsonl \
  --benchmark_jsonl data/benchmark/himalaya_indic_test.jsonl \
  --max_samples 30000 2>&1 | tee prep.log

# Merge into train/val (2% val split for training monitoring)
python scripts/prepare/merge.py \
  --interim_dir data/interim \
  --out_dir data/processed --val_ratio 0.02

# Aggregate held-out benchmarks
HELDOUT=data/benchmark/heldout_test.jsonl
: > "$HELDOUT"
shopt -s nullglob
for f in data/benchmark/*_test.jsonl; do
    [ "$f" = "$HELDOUT" ] && continue
    cat "$f" >> "$HELDOUT"
done

wc -l data/processed/train.jsonl data/processed/val.jsonl data/benchmark/heldout_test.jsonl
# Expected: train ~29,400, val ~600, heldout ~600
```

Detach with **Ctrl+B then D** while prep runs.

---

## Step 4 — Train iter-3 (~4-8 hrs, ~$15-25)

Once prep completes:

```bash
tmux new -s iter3

# Re-export
export HF_TOKEN=<NEW-rotated-hf-token>
export WANDB_API_KEY=<NEW-rotated-wandb-key>
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd ~/enclave-scribe && source .venv/bin/activate

# Launch
torchrun --nproc_per_node=4 scripts/train.py \
  --config configs/train/iter3.yaml 2>&1 | tee train.log
```

**Detach after you see the first loss line at step 20** (~5 min). Check on W&B live: https://wandb.ai/

---

## Step 5 — Eval iter-3 vs iter-1 vs base on same held-out (~1-2 hrs, ~$5)

Once training completes (final `train_loss` printed):

```bash
# Sync checkpoint to S3
aws s3 sync outputs/iter3/ s3://enclave-scribe-checkpoints/outputs/iter3/

# Pull iter-1 adapter for comparison
mkdir -p outputs/iter1
aws s3 sync s3://enclave-scribe-checkpoints/outputs/iter1/ outputs/iter1/ \
  --exclude "checkpoint-*/*"

# Sample 500 stratified from heldout for eval (avoid the "first 500 = one category" iter-2 bug)
python -c "
import json, random
random.seed(42)
lines = open('data/benchmark/heldout_test.jsonl').readlines()
random.shuffle(lines)
open('data/benchmark/heldout_500.jsonl','w').writelines(lines[:500])
print(f'Sampled 500 from {len(lines)} total')
"

# Run all 3 evals (Iter-3 first — the one we care about most)
for adapter_dir in outputs/iter3 outputs/iter1 ""; do
  name=$(basename ${adapter_dir:-base})
  echo "=== Evaluating $name ==="
  if [ -n "$adapter_dir" ]; then
    python scripts/eval.py --gt_jsonl data/benchmark/heldout_500.jsonl \
      --image_root data/raw --base_model allenai/olmOCR-2-7B-1025 \
      --adapter_dir $adapter_dir --out_json results/${name}_heldout500.json
  else
    python scripts/eval.py --gt_jsonl data/benchmark/heldout_500.jsonl \
      --image_root data/raw --base_model allenai/olmOCR-2-7B-1025 \
      --out_json results/base_heldout500.json
  fi
done

# Also eval on your reference gazette PDF (real-world qualitative test)
python scripts/agent/parse.py \
  --pdf ~/gazette.pdf \
  --out results/iter3_gazette.md \
  --base_model allenai/olmOCR-2-7B-1025 \
  --adapter_dir outputs/iter3

# Sync results
aws s3 sync results/ s3://enclave-scribe-checkpoints/results/iter3/
```

---

## Step 6 — Stop instance to end billing

```bash
sudo shutdown -h now
```

Then from your local machine:
```bash
INSTANCE_ID=$(cat /tmp/iter3_instance_id.txt)
aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region us-east-1
```

---

## Success criteria for iter-3

Iter-3 "succeeded" if any of these hold on the held-out eval:

- **Iter-3 CER < iter-1 CER** on Devanagari / Indic samples (measurable via `heldout_500.json`)
- **Iter-3 markdown output on your reference gazette** produces Unicode Devanagari that matches your `output.md` reference more closely than OLMoCR-2-baseline did in Phase 0
- **Iter-3 doesn't regress on English** (English CER within 5% of iter-1)

If iter-3 wins on Indic without regressing on English → we ship it, publish, iter-4 focuses on scaling data (add SynthIndic, more real gazettes) and image understanding (seals, charts).

If iter-3 regresses on English → we're overfitting to Indic. Reduce mix ratio in iter-4.

If iter-3 doesn't beat iter-1 on Indic → data is too weak. Look at augmenting with LlamaParse-labeled real gazettes.

---

## Reminders

- **Never touch staging instance** `i-073a0fe419ceb9f49` — it's your production staging env
- **Rotate keys after any session** where they were pasted anywhere
- **Terminate, don't just stop** — stopped instances still bill for EBS (~$40/mo)
- **Reserve stratified test set** — never eval on `head -N` of held-out; always sample
