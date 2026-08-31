# Iter-4 runbook — page-level Devanagari fine-tune

Copy-paste sequence to run iter-4 on AWS. Builds on iter-3 (continues from
its adapter). All heavy work runs on a fresh g5.xlarge; local machine only
launches / terminates the instance.

**Objective**: fix iter-3's weakness on full-page structure by training on
~3,000 page-level Devanagari samples (IndicVisionBench, NayanaBench, etc.)
loaded via `himalaya-ai/devanagari_ocr_dataset`.

**Budget expectation**: ~$8-10 compute (6-8 hrs training on g5.xlarge on-demand)
+ ~$3 setup/eval = ~$13 total.

**Success criteria**:
- iter-4 CER on page-level held-out < iter-3 CER on same held-out
- iter-4 does NOT regress on the word-level held-out from iter-3
- English CER on reserved CORD/FUNSD held-out within 5% of base (no catastrophic forgetting)
- `parse.py` on gazette.pdf runs cleanly WITHOUT `bad_words_ids` (the workaround)

---

## Precondition — rotate keys if pasted last session

- HuggingFace token at [hf.co/settings/tokens](https://hf.co/settings/tokens) → delete old, create new **write** token
- W&B API key at [wandb.ai](https://wandb.ai) → Settings → revoke, create new

---

## Step 1 — Launch g5.xlarge (~5 min)

On your local Mac:

```bash
INSTANCE_ID=$(aws ec2 run-instances \
  --image-id ami-012ba162b9cd2729c \
  --instance-type g5.xlarge \
  --key-name enclave-scribe-key \
  --security-group-ids $(aws ec2 describe-security-groups \
    --filters "Name=group-name,Values=enclave-scribe-sg" \
    --region us-east-1 --query "SecurityGroups[0].GroupId" --output text) \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":500,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
  --count 1 --region us-east-1 \
  --query "Instances[0].InstanceId" --output text)
echo "$INSTANCE_ID" > /tmp/iter4_instance_id.txt

aws ec2 wait instance-running --instance-ids $INSTANCE_ID --region us-east-1
PUBLIC_IP=$(aws ec2 describe-instances --instance-ids $INSTANCE_ID \
  --region us-east-1 --query "Reservations[0].Instances[0].PublicIpAddress" --output text)
echo "IP: $PUBLIC_IP"

# Upload the reference gazette for the qualitative test at the end
scp -i enclave-scribe-key.pem \
  '/Users/alias1623/Downloads/36_SO3977(E)_14082018.pdf' \
  ubuntu@$PUBLIC_IP:~/gazette.pdf
ssh -i enclave-scribe-key.pem ubuntu@$PUBLIC_IP
```

---

## Step 2 — Instance setup (~5 min)

```bash
cd ~
git clone https://github.com/Enclave-Labs-Inc/enclave-scribe.git
cd enclave-scribe

source /opt/pytorch/bin/activate
pip install -e . editdistance accelerate peft

# Rotated tokens
export HF_TOKEN=<NEW-hf-write-token>
export WANDB_API_KEY=<NEW-wandb-key>
hf auth login --token $HF_TOKEN

# AWS creds for S3 pull/push
aws configure
# region us-east-1, format json

# Verify GPU + libraries
python -c "
import torch, transformers, peft
print(f'torch {torch.__version__}, cuda={torch.cuda.is_available()}, gpus={torch.cuda.device_count()}')
print(f'transformers {transformers.__version__}, peft {peft.__version__}')
"
# Expect: cuda=True, gpus=1
```

---

## Step 3 — Pull iter-3 adapter (~30 sec)

Iter-4 continues from iter-3's LoRA, so the adapter must exist at `outputs/iter3/`.

```bash
mkdir -p outputs/iter3
aws s3 sync s3://enclave-scribe-checkpoints/outputs/iter3/ outputs/iter3/ \
  --exclude "checkpoint-*/*"
ls -lh outputs/iter3/
# Expect adapter_model.safetensors (~380 MB) + adapter_config.json
```

---

## Step 4 — Data prep (~1-2 hrs, ~50 GB disk)

```bash
tmux new -s prep

export HF_TOKEN=<NEW-hf-write-token>
cd ~/enclave-scribe && source /opt/pytorch/bin/activate

# First run: auto-diagnoses the prefix distribution BEFORE filtering.
# Watch the "PREFIX DISTRIBUTION" table it prints. If the default
# --include_prefixes don't match your data, rerun with the right ones.
python scripts/prepare/prep_himalaya_pages.py \
  --raw_dir data/raw \
  --out_jsonl data/interim/himalaya_pages.jsonl \
  --benchmark_jsonl data/benchmark/himalaya_pages_test.jsonl \
  --max_samples 3000 2>&1 | tee prep.log

# Aggregate train/val
python scripts/prepare/merge.py \
  --interim_dir data/interim \
  --out_dir data/processed --val_ratio 0.05

# Rebuild the held-out aggregate
HELDOUT=data/benchmark/heldout_test.jsonl
: > "$HELDOUT"
shopt -s nullglob
for f in data/benchmark/*_test.jsonl; do
    [ "$f" = "$HELDOUT" ] && continue
    cat "$f" >> "$HELDOUT"
done

wc -l data/processed/train.jsonl data/processed/val.jsonl data/benchmark/heldout_test.jsonl
```

Detach with **Ctrl+B then D** while prep runs.

---

## Step 5 — Pull English held-out for regression check (~1 min)

```bash
# Iter-2 reserved CORD + FUNSD test splits. We reuse them as our English
# regression benchmark for iter-4.
mkdir -p data/benchmark
aws s3 cp s3://enclave-scribe-checkpoints/results/iter2/iter2_heldout500.json /tmp/
python -c "
import json
# The iter-2 result file has predictions AND ground truth per sample.
# Extract ground truth into a benchmark JSONL.
with open('/tmp/iter2_heldout500.json') as f:
    results = json.load(f)
with open('data/benchmark/english_test.jsonl', 'w') as f:
    for r in results.get('results') or results.get('per_sample') or []:
        cat = r.get('category', '')
        if any(c in cat.lower() for c in ('cord', 'funsd', 'docvqa', 'english')):
            f.write(json.dumps({
                'image':    r.get('image', ''),
                'text':     r.get('gt', r.get('text', '')),
                'category': cat,
            }, ensure_ascii=False) + '\n')
print('Wrote data/benchmark/english_test.jsonl')
"
wc -l data/benchmark/english_test.jsonl
head -1 data/benchmark/english_test.jsonl | python -m json.tool
# NOTE: images referenced by english_test.jsonl live under data/raw/ and
# will NOT be present on this fresh instance. Either sync them from S3
# (if archived) or skip the English eval and rely on the qualitative
# gazette test for English regression signal.
```

---

## Step 6 — Train iter-4 (~6-8 hrs, ~$8)

```bash
tmux new -s iter4
export HF_TOKEN=<NEW-hf-write-token>
export WANDB_API_KEY=<NEW-wandb-key>
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd ~/enclave-scribe && source /opt/pytorch/bin/activate

python scripts/train.py --config configs/train/iter4.yaml 2>&1 | tee train.log
```

Watch for `"Resuming LoRA from adapter: outputs/iter3"` in the first ~30 sec
of the run. That confirms `resume_adapter` took effect.

Detach after seeing the first loss line (~5 min in). Track live on wandb.

---

## Step 7 — Eval iter-4 (~30 min, ~$0.50)

```bash
# Sync adapter to S3 first (protect against instance loss)
aws s3 sync outputs/iter4/ s3://enclave-scribe-checkpoints/outputs/iter4/

# Sample 500 stratified from the pages held-out
python -c "
import json, random
random.seed(42)
lines = open('data/benchmark/himalaya_pages_test.jsonl').readlines()
random.shuffle(lines)
open('data/benchmark/pages_500.jsonl', 'w').writelines(lines[:500])
print(f'Sampled 500 from {len(lines)}')
"

# iter-4 vs iter-3 vs base on pages held-out
for adapter in outputs/iter4 outputs/iter3 ""; do
  name=$(basename ${adapter:-base})
  echo "=== $name ==="
  if [ -n "$adapter" ]; then
    python scripts/eval.py \
      --gt_jsonl data/benchmark/pages_500.jsonl \
      --image_root data/raw \
      --base_model allenai/olmOCR-2-7B-1025 \
      --adapter_dir $adapter \
      --out_json results/${name}_pages500.json
  else
    python scripts/eval.py \
      --gt_jsonl data/benchmark/pages_500.jsonl \
      --image_root data/raw \
      --base_model allenai/olmOCR-2-7B-1025 \
      --out_json results/${name}_pages500.json
  fi
done

# Real-world qualitative test — should work WITHOUT bad_words_ids workaround
python scripts/agent/parse.py \
  --pdf ~/gazette.pdf \
  --out results/iter4_gazette.md \
  --base_model allenai/olmOCR-2-7B-1025 \
  --adapter_dir outputs/iter4

# Sync results
aws s3 sync results/ s3://enclave-scribe-checkpoints/results/iter4/
```

---

## Step 8 — Terminate

```bash
sudo shutdown -h now
```

From local:

```bash
INSTANCE_ID=$(cat /tmp/iter4_instance_id.txt)
aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region us-east-1
```

---

## Reminders

- **Never touch staging instance** `i-073a0fe419ceb9f49`
- **Rotate keys** after this session
- **Terminate, don't stop** — stopped instances still bill for EBS
- **If prep script's PREFIX DISTRIBUTION shows unexpected prefixes**, rerun with `--include_prefixes <right_prefixes>` — the defaults are best-guess
- **If OOM during training**, drop `max_length: 8192` to `6144` in iter4.yaml
