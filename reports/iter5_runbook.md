# Iter-5 runbook — 32B LoRA on olmOCR-2-32B-1025

Copy-paste sequence for iter-5 training. Follows the same shape as
`reports/iter4_runbook.md`. Assumes the [iter-5 plan](../.claude/plans/vectorized-dreaming-hearth.md)
and the [dry-run findings](iter5/DRYRUN.md) have been read.

**Objective**: does a bigger base model absorb iter-4's pseudo-label
corpus better than 7B did? Iter-4's flat training loss (~4.2 across 82
steps) suggested a capacity ceiling. Iter-5 tests that on 32B.

**Ship gate (all four must pass, from the plan file)**:
1. Training loss drops below 4.0 by step 40; ends < 3.5
2. Gazette gate passes (see `tests/fixtures/pdfs/README.md`)
3. **Devanagari word CER ≤ 21.5%** on `data/benchmark/himalaya_500.jsonl` (updated 2026-09-10 after iter-4's dry-run measured 28.1% — regression from iter-3)
4. Head-to-head gazette chars: iter-5 ≥ iter-4's 13,646

**Budget expectation**: $50-100 depending on instance choice.

---

## Precondition — AWS quota

At time of writing (2026-09-11), account 530456407613 has:
- P instance on-demand: 0 vCPU quota
- P Spot: 0 vCPU quota
- G instance on-demand: 64 vCPU (only fits g5.16xlarge or smaller)

Iter-5 needs one of:
- **p4de.24xlarge** (96 vCPU P quota) — 8× A100 80GB, ~$32.77/hr on-demand, ~$12/hr spot
- **g5.48xlarge** (192 vCPU G quota) — 8× A10G 24GB, ~$16.29/hr on-demand

Quota request steps: Service Quotas → EC2 → us-east-1, request 96 P (both spot + on-demand) AND 200 G (both spot + on-demand). Justification text lives in the plan file. Typical wait: 24-72h.

Check what landed before proceeding:
```bash
aws ec2 run-instances --dry-run --region us-east-1 \
    --image-id ami-012ba162b9cd2729c \
    --instance-type p4de.24xlarge \
    --key-name enclave-scribe-key \
    --security-group-ids sg-04fe5a45cf7ad4a53 \
    --count 1 2>&1 | tail -2
# "would have succeeded" = quota OK
# "VcpuLimitExceeded" or "MaxSpotInstanceCountExceeded" = still blocked
```

## Precondition — HF token

`hf_YjDSR...` may have been rotated after the dry-run. If so, get a new **write** token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

## Precondition — do NOT touch staging

`i-073a0fe419ceb9f49` (g5.xlarge in us-east-1) is the staging environment. All commands below must never terminate or modify it. Instance IDs are quoted verbatim in every destructive step for safety.

---

## Step 1 — Launch (choose ONE variant)

### 1a. If P quota landed → p4de.24xlarge (preferred)

```bash
# Local Mac:
SG_ID=sg-04fe5a45cf7ad4a53
AMI=ami-012ba162b9cd2729c
KEY=enclave-scribe-key
UD=/private/tmp/claude-501/*/*/scratchpad/iter5_p4de_userdata.sh  # or reuse iter4's launch userdata

# Spot preferred (~$12/hr vs $32.77/hr on-demand):
aws ec2 run-instances \
    --region us-east-1 \
    --image-id "$AMI" \
    --instance-type p4de.24xlarge \
    --key-name "$KEY" \
    --security-group-ids "$SG_ID" \
    --instance-market-options 'MarketType=spot' \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":500,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
    --user-data "file://$UD" \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=iter5-training}]' \
    --count 1
```

Config: `configs/train/iter5.yaml`. Launch: `torchrun --nproc_per_node=8`.

### 1b. If only G quota landed → g5.48xlarge (fallback)

Same command as (1a), swap `--instance-type g5.48xlarge`. Config: `configs/train/iter5_g5.yaml` (max_pixels lowered, LoRA r=32, grad_accum=4).

**Rollback if LoRA+FSDP crashes at step 0 on A10G**: remove `fsdp` + `fsdp_config` keys from the yaml, restart. Base model may not fit — see the config's rollback comment.

---

## Step 2 — Setup verification (~10-15 min after launch)

```bash
IP=$(aws ec2 describe-instances --region us-east-1 --instance-ids <INSTANCE_ID> \
    --query 'Reservations[].Instances[].PublicIpAddress' --output text)

# Poll setup completion
ssh -i enclave-scribe-key.pem ubuntu@$IP "grep -c 'Setup complete' setup.log"
# Should return 1 once cloud-init finishes deps install
```

Once setup done, configure AWS creds + HF token (repeat the pattern from `reports/iter4_runbook.md` since IAM cred-transfer via bash is blocked by the tool classifier):

```bash
ssh -i enclave-scribe-key.pem ubuntu@$IP
# On instance:
aws configure                              # paste keys
echo 'export HF_TOKEN=<HF_WRITE_TOKEN>' >> ~/.bashrc
export HF_TOKEN=<HF_WRITE_TOKEN>
huggingface-cli login --token "$HF_TOKEN"

# Verify GPU + cuDNN
source ~/enclave-scribe/.venv/bin/activate
python -c "import torch; print(torch.cuda.device_count(), 'GPUs; CUDA:', torch.cuda.is_available())"
# p4de → 8; g5.48xlarge → 8
```

---

## Step 3 — Sync corpus + adapters from S3 (~2 min)

Iter-5 uses iter-4's corpus verbatim (793 pseudo-labeled pages + 500 word replay).

```bash
cd ~/enclave-scribe
mkdir -p data/processed outputs

aws s3 sync s3://enclave-scribe-checkpoints/data/iter4/processed/ data/processed/
aws s3 sync s3://enclave-scribe-checkpoints/outputs/iter3/ outputs/iter3/ --exclude "checkpoint-*"
aws s3 sync s3://enclave-scribe-checkpoints/outputs/iter4/ outputs/iter4/ --exclude "checkpoint-*"

wc -l data/processed/train.jsonl data/processed/val.jsonl
# Expect 1254 train + 42 val
```

For image files, `data/processed/train.jsonl` references relative paths under `data/raw/indicdlp_pages/` and `data/raw/himalaya_indic/`. Regenerate them from HF (both are large — allow disk + time):

```bash
# Bug-history reference: prep_himalaya_indic first hit an OOM on 16GB boxes.
# 62GB+ RAM on p4de/g5.48xlarge is comfortable.
python scripts/prepare/prep_indicdlp_pages.py \
    --raw_dir data/raw/indicdlp_pages \
    --manifest_jsonl data/interim/indicdlp_pages.manifest.jsonl \
    --max_per_lang 1800
python scripts/prepare/prep_himalaya_indic.py \
    --raw_dir data/raw --out_jsonl data/interim/himalaya_indic.jsonl \
    --max_samples 500
```

---

## Step 4 — Kick off training

```bash
# Ensure LD_LIBRARY_PATH is set (baked into UserData bashrc; verify)
source ~/.bashrc
echo $LD_LIBRARY_PATH | grep cudnn && echo OK

# Launch in tmux so ssh drop doesn't kill training
tmux new-session -d -s train
tmux send-keys -t train "cd ~/enclave-scribe && source .venv/bin/activate" Enter

# For p4de (iter5.yaml, uses 4 GPUs first per plan's rollback strategy):
tmux send-keys -t train "torchrun --nproc_per_node=4 scripts/train.py --config configs/train/iter5.yaml 2>&1 | tee train.log" Enter

# For g5.48xlarge (iter5_g5.yaml, 8 GPUs):
# tmux send-keys -t train "torchrun --nproc_per_node=8 scripts/train.py --config configs/train/iter5_g5.yaml 2>&1 | tee train.log" Enter

tmux send-keys -t train "aws s3 sync outputs/iter5/ s3://enclave-scribe-checkpoints/outputs/iter5/ --quiet" Enter
```

**First 5-10 min watch**: look for the "Loading checkpoint shards" progress (32B is ~64GB, ~15-30 sec per shard). Once training starts, log lines every 5 steps.

## Step 5 — Early kill criterion (per plan)

**Kill by step 40 if training loss stays flat at ~4.2** (iter-4's failure mode):
```bash
# From local Mac:
ssh -i enclave-scribe-key.pem ubuntu@$IP "grep -E \"'loss':\" ~/enclave-scribe/train.log | tail -20"
```

If loss doesn't drop below 4.0 by step 40, kill training and treat as postmortem (pseudo-label ceiling confirmed):
```bash
ssh -i enclave-scribe-key.pem ubuntu@$IP "tmux send-keys -t train C-c"
# Then jump to Step 8 termination.
```

Otherwise continue.

---

## Step 6 — Ship gate

### 6a. Sync adapter to S3 (protect against instance loss)

```bash
aws s3 sync outputs/iter5/ s3://enclave-scribe-checkpoints/outputs/iter5/ --quiet
```

### 6b. Devanagari word CER (must be ≤ 21.5%)

```bash
python scripts/eval.py \
    --gt_jsonl    data/benchmark/himalaya_500.jsonl \
    --image_root  data/raw \
    --base_model  allenai/olmOCR-2-32B-1025 \
    --adapter_dir outputs/iter5 \
    --out_json    results/iter5/iter5_devanagari.json \
    --max_new_tokens 64

python -c "
import json
r = json.load(open('results/iter5/iter5_devanagari.json'))
cer = r['overall']['cer']
print(f'iter-5 Devanagari CER: {cer:.4f}')
assert cer <= 0.215, f'REGRESSION: iter-5 CER {cer:.4f} > gate 0.215'
print('PASS: within gate')
"
```

### 6c. Gazette gate

Ensure `bad_words_ids` is DISABLED in `scribe/agent/tools.py::extract_page` (comment out line 95):
```bash
grep -n bad_words_ids scribe/agent/tools.py
sed -i 's|.*bad_words_ids=bad_words_ids or None,.*|# &  # temporarily disabled for iter-5 ship gate|' scribe/agent/tools.py
```

Then run:
```bash
python scripts/agent/parse.py \
    --pdf         tests/fixtures/pdfs/gazette_moef_2024_06_07.pdf \
    --out         results/iter5/gazette_iter5.md \
    --base_model  allenai/olmOCR-2-32B-1025 \
    --adapter_dir outputs/iter5

# Verify the 5 gate criteria from tests/fixtures/pdfs/README.md
grep -c "<tool_call>" results/iter5/gazette_iter5.md   # must be 0
wc -c results/iter5/gazette_iter5.md                   # must be >= 13646 (iter-4 baseline)
```

Revert the tools.py edit after:
```bash
git checkout scribe/agent/tools.py
```

### 6d. Cross-adapter regression harness

```bash
python scripts/eval_regression.py \
    --base_model       allenai/olmOCR-2-32B-1025 \
    --adapter          iter3:outputs/iter3 \
    --adapter          iter4:outputs/iter4 \
    --adapter          iter5:outputs/iter5 \
    --devanagari_jsonl data/benchmark/himalaya_500.jsonl \
    --gazette_pdf      tests/fixtures/pdfs/gazette_moef_2024_06_07.pdf \
    --out              results/iter5/regression_table.md \
    --devanagari_max_new_tokens 64
```

**Note**: `--base_model` mismatch between iter-3/iter-4 (7B) and iter-5 (32B) means those two adapters won't load onto the 32B base — expect them to error out or produce garbage. Comparison table below still works if we skip iter-3/iter-4 in this final table and cite their prior numbers from `reports/iter5/DRYRUN.md`.

---

## Step 7 — If all gates pass: publish + report

```bash
# Publish adapter to HF
python scripts/publish_hf.py \
    --adapter_dir outputs/iter5 \
    --repo_id    enclavelabs/enclave-scribe-devanagari-iter5 \
    --model_card reports/iter5/MODEL_CARD.md

# Sync all iter-5 artifacts
aws s3 sync results/iter5/ s3://enclave-scribe-checkpoints/results/iter5/ --quiet
```

Then locally: write `reports/iter5/README.md` (same shape as `reports/iter4/README.md`), open PR.

---

## Step 8 — Terminate the training instance

**READ THE INSTANCE ID BEFORE PRESSING ENTER. Never `i-073a0fe419ceb9f49` (staging).**

```bash
# Local Mac:
TARGET=<iter-5 instance ID from Step 1>
STAGING=i-073a0fe419ceb9f49
[ "$TARGET" = "$STAGING" ] && echo "ABORT" && exit 1
aws ec2 terminate-instances --region us-east-1 --instance-ids $TARGET

# Verify both:
aws ec2 describe-instances --region us-east-1 \
    --instance-ids $TARGET $STAGING \
    --query 'Reservations[].Instances[].[InstanceId,State.Name]' --output table
# Expected: $TARGET → shutting-down, $STAGING → running
```

---

## If iter-5 fails the ship gate

Do NOT publish. Write `reports/iter5/POSTMORTEM.md` explaining which gate failed and why. Next-iter options at that point are:
1. **Retrain with different LR/epochs** on same corpus (cheap, ~$50)
2. **Escalate to human labels** (iter-6 pilot in `reports/iter6/PILOT.md`)
3. **Try even bigger base** — Qwen2.5-VL-72B, olmOCR-2-72B if AWS quota allows

The postmortem should pick one and open the iter-6 PR.
