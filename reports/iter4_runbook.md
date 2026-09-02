# Iter-4 runbook — page-level Devanagari via hybrid bootstrap

Copy-paste sequence to run iter-4 on AWS. Builds on iter-3 (continues from
its adapter). All heavy work runs on a fresh g5.xlarge; local Mac only
launches and terminates.

**Objective**: fix iter-3's weakness on full-page structure so the runtime
`bad_words_ids` workaround in `scribe/agent/tools.py::extract_page` (PR #41)
becomes redundant.

**Data strategy (revised 2026-09-02)**: himalaya-ai turned out to have no
page-level content. New plan uses `ai4bharat/indicdlp` for real page
images + iter-3 as pseudo-labeler. See PLAN for the discovery story.

**Budget expectation**: ~$60-160 total. Cost is dominated by bulk
pseudo-labeling (~$50-150 for 3k pages at iter-3's ~1-3 min/page).
Training itself is ~$5-7.

**Ship gate (single unambiguous test)**: rerun `parse.py` on the reference
gazette.pdf with `bad_words_ids` DISABLED in `extract_page`. iter-4 ships
if the output is clean (~25k chars across 9 pages, no page at exactly
`max_new_tokens=4096` ceiling, `grep -c '<tool_call>'` = 0).

---

## Precondition — rotate keys

- HF write token at [hf.co/settings/tokens](https://hf.co/settings/tokens) — revoke old, create new **write** token
- W&B key at [wandb.ai](https://wandb.ai) → Settings → revoke, create new
- **Accept IndicDLP gated terms**: visit https://huggingface.co/datasets/ai4bharat/indicdlp — click "Agree and access repository" while logged in with the account matching your HF token

---

## Gate 0 — IndicDLP image quality sanity check (~5 min, ~$0)

**On local Mac** (no GPU needed for this): pull 20 samples and eyeball.

```bash
cd ~/enclave/enclave-scribe
python3 -c "
from datasets import load_dataset
from PIL import Image
import os
os.makedirs('/tmp/indicdlp_sanity', exist_ok=True)
ds = load_dataset('ai4bharat/indicdlp', split='train', streaming=True, token='<YOUR-HF-TOKEN>')
saved = {'hi': 0, 'mr': 0}
for i, s in enumerate(ds):
    lang = None
    for k in ('language','lang','lang_code'):
        if k in s:
            lang = str(s[k]).lower(); break
    if lang not in ('hi','mr') or saved[lang] >= 10: continue
    img = s.get('image') or s.get('img') or s.get('png')
    if img is None: continue
    img.convert('RGB').save(f'/tmp/indicdlp_sanity/{lang}_{saved[lang]:02d}.png')
    saved[lang] += 1
    if sum(saved.values()) >= 20: break
print(saved)
print('saved to /tmp/indicdlp_sanity/')
"
open /tmp/indicdlp_sanity
```

**Decision**:
- ✅ Text is clearly readable at native resolution → proceed to Step 1
- ❌ Text is too small / blurry → ABORT this plan, pivot to manual gazette download (see fallback section at bottom)

---

## Step 1 — Launch g5.xlarge (~5 min)

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

# Upload reference gazette for the ship-gate test at the end
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
pip install -e . editdistance accelerate peft datasets wandb liger-kernel

# Tokens
export HF_TOKEN=<NEW-hf-write-token>
export WANDB_API_KEY=<NEW-wandb-key>
hf auth login --token $HF_TOKEN

# AWS creds
aws configure   # region us-east-1, format json

# Verify
python -c "
import torch, transformers, peft
print(f'torch {torch.__version__}, cuda={torch.cuda.is_available()}, gpus={torch.cuda.device_count()}')
print(f'transformers {transformers.__version__}, peft {peft.__version__}')
"
```

---

## Step 3 — Pull iter-3 adapter (~30 sec)

```bash
mkdir -p outputs/iter3
aws s3 sync s3://enclave-scribe-checkpoints/outputs/iter3/ outputs/iter3/ \
  --exclude "checkpoint-*/*"
ls -lh outputs/iter3/
# Expect adapter_model.safetensors (~380 MB) + adapter_config.json
```

---

## Phase 1 — Download IndicDLP page images (~40 min, ~$0.50)

`prep_indicdlp_pages.py` downloads one parquet shard at a time,
detects its language from the first row's image path, keeps matches
(extracts all images to disk), deletes non-matches, and stops walking
a language's shard block once the language boundary is hit. Known
first-shard indices for hi and mr are baked in from a 2026-09-02
probe (see `LANG_START_SHARD` in the script).

```bash
tmux new -s prep1
source /opt/pytorch/bin/activate

export HF_TOKEN=<NEW-hf-write-token>   # re-export in the tmux shell

python scripts/prepare/prep_indicdlp_pages.py \
  --raw_dir data/raw/indicdlp_pages \
  --manifest_jsonl data/interim/indicdlp_pages.manifest.jsonl \
  --langs hi mr \
  --max_per_lang 1800 2>&1 | tee prep1.log

wc -l data/interim/indicdlp_pages.manifest.jsonl
du -sh data/raw/indicdlp_pages/
```

Detach with **Ctrl+B then D**. Expected download: ~13 shards for hi + ~13
for mr = ~26 shards × ~500 MB downloaded (auto-deleted after extraction).
Peak disk usage during Phase 1: ~1 GB.

If a shard's language doesn't match its expected position, the script
prints a clear boundary message and moves on. If a target language is
unknown (not in LANG_START_SHARD), the script errors with instructions.

---

## Phase 2 — Bulk pseudo-label (~50-150 GPU-hrs, ~$50-150)

**This is the expensive step.** User approved committing upfront. Runs
serial iter-3 inference per page.

```bash
tmux new -s prep2
source /opt/pytorch/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python scripts/prepare/label_pages_with_iter3.py \
  --manifest_jsonl data/interim/indicdlp_pages.manifest.jsonl \
  --raw_dir data/raw/indicdlp_pages \
  --out_jsonl data/interim/indicdlp_labeled.jsonl \
  --base_model allenai/olmOCR-2-7B-1025 \
  --adapter_dir outputs/iter3 2>&1 | tee prep2.log
```

Script is resumable — safe to detach, and if the instance dies, on rerun
it skips images already in `indicdlp_labeled.jsonl`.

**Sync labels to S3 every few hours** to protect the investment:
```bash
# In another SSH session
aws s3 sync data/interim/ s3://enclave-scribe-checkpoints/data/iter4/interim/
```

**When Phase 2 completes, spot-check 20 labels manually**:
```bash
python -c "
import json, random
random.seed(0)
lines = [l for l in open('data/interim/indicdlp_labeled.jsonl') if json.loads(l).get('filter') == 'ok']
random.shuffle(lines)
for l in lines[:20]:
    r = json.loads(l)
    print('---')
    print(f'{r[\"image\"]} ({r[\"language\"]})')
    print(r['text'][:400])
"
```
If most look like coherent Hindi/Marathi document text → proceed to Phase 4.
If most are garbage → tighten filters and rerun `--filter` step (add
`--min_chars 200` or similar).

---

## Phase 4 — Assemble training corpus (~5 min, ~$0)

Word replay first (need ~500 word-level samples in the mix). Uses the
same himalaya-ai prep script we shipped for iter-3:

```bash
python scripts/prepare/prep_himalaya_indic.py \
  --raw_dir data/raw \
  --out_jsonl data/interim/himalaya_indic.jsonl \
  --max_samples 500
```

Then build the corpus:

```bash
python scripts/prepare/build_iter4_corpus.py \
  --labeled_pages_jsonl data/interim/indicdlp_labeled.jsonl \
  --word_replay_jsonl data/interim/himalaya_indic.jsonl \
  --train_out data/processed/train.jsonl \
  --val_out data/processed/val.jsonl \
  --max_pages 2500 \
  --max_word_replay 500 \
  --val_ratio 0.05

wc -l data/processed/train.jsonl data/processed/val.jsonl
```

Expect ~2400 train (2375 pages + 500 words - rounding) and ~125 val.

Sync the corpus to S3 too:
```bash
aws s3 sync data/processed/ s3://enclave-scribe-checkpoints/data/iter4/processed/
```

---

## Phase 5 — Train iter-4 (~5-7 hrs, ~$7)

```bash
tmux new -s iter4
source /opt/pytorch/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_TOKEN=<NEW-hf-write-token>
export WANDB_API_KEY=<NEW-wandb-key>

python scripts/train.py --config configs/train/iter4.yaml 2>&1 | tee train.log
```

Watch for `"Resuming LoRA from adapter: outputs/iter3"` in the first
~30 seconds. Also verify starting loss is reasonable (~2-4, NOT ~7 like
fresh training — that would mean resume didn't take).

---

## Phase 6 — Verify + ship gate (~30 min, ~$1)

### Sync the adapter first (protect against instance loss)

```bash
aws s3 sync outputs/iter4/ s3://enclave-scribe-checkpoints/outputs/iter4/
```

### Word regression bonus check — iter-4 must not regress on iter-3's win

```bash
# Pull iter-3's held-out word set from S3
mkdir -p data/benchmark
aws s3 cp s3://enclave-scribe-checkpoints/results/iter3/iter3_heldout500.json /tmp/
python -c "
import json
res = json.load(open('/tmp/iter3_heldout500.json'))
# Adapt shape — iter-3 result file has per-sample records with image+gt
per = res.get('per_sample') or res.get('results') or []
with open('data/benchmark/iter3_words_500.jsonl', 'w') as f:
    for r in per:
        f.write(json.dumps({
            'image':    r.get('image', ''),
            'text':     r.get('gt', r.get('text', '')),
            'category': 'himalaya_indic',
        }, ensure_ascii=False) + '\n')
print('Wrote', sum(1 for _ in open('data/benchmark/iter3_words_500.jsonl')))
"

# Eval iter-4 on the same word set
python scripts/eval.py \
  --gt_jsonl data/benchmark/iter3_words_500.jsonl \
  --image_root data/raw \
  --base_model allenai/olmOCR-2-7B-1025 \
  --adapter_dir outputs/iter4 \
  --out_json results/iter4_words500.json

# Compare CER — must be within 5% of iter-3's 0.174
python -c "
import json
iter3 = 0.174  # from iter-3 report
iter4 = json.load(open('results/iter4_words500.json'))['per_category']['himalaya_indic']['cer']
print(f'iter-3 CER: {iter3}')
print(f'iter-4 CER: {iter4}')
print(f'delta:      {iter4 - iter3:+.4f}')
print('PASS' if iter4 <= iter3 * 1.05 else 'FAIL — word regression')
"
```

### SHIP GATE — gazette.pdf without the bad_words_ids workaround

Temporarily disable the workaround in `scribe/agent/tools.py` (git-tracked
line — do NOT commit this change, revert after the test):

```bash
# Comment out bad_words_ids in extract_page
sed -i 's/bad_words_ids=bad_words_ids or None,/# bad_words_ids=bad_words_ids or None,  # DISABLED for iter-4 ship gate/' scribe/agent/tools.py
grep -n "bad_words_ids" scribe/agent/tools.py

# Run parse on gazette
python scripts/agent/parse.py \
  --pdf ~/gazette.pdf \
  --out results/iter4_gazette_no_workaround.md \
  --base_model allenai/olmOCR-2-7B-1025 \
  --adapter_dir outputs/iter4 2>&1 | tee results/iter4_parse.log

# Ship-gate checks
grep "chars in" results/iter4_parse.log
grep -c "<tool_call>" results/iter4_gazette_no_workaround.md
wc -c results/iter4_gazette_no_workaround.md
```

**PASS** if:
- All 9 pages produce non-zero, non-looping content (no page at exactly ~240s)
- `grep -c "<tool_call>"` returns 0
- Total chars ≥ 25k

**Revert the sed change so we don't accidentally commit it**:
```bash
git checkout scribe/agent/tools.py
grep -n "bad_words_ids" scribe/agent/tools.py    # confirm it's back
```

### Sync all results

```bash
aws s3 sync results/ s3://enclave-scribe-checkpoints/results/iter4/
```

---

## Step 7 — Terminate

```bash
sudo shutdown -h now
```

From local Mac:

```bash
INSTANCE_ID=$(cat /tmp/iter4_instance_id.txt)
aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region us-east-1
```

---

## Fallback — if Gate 0 fails (IndicDLP images unreadable)

Pivot to manual gazette PDF download:

1. Download 50-100 real Indian gazette PDFs from https://egazette.gov.in/ (browse by ministry / date)
2. Upload directory to instance: `scp -r ~/gazettes/ ubuntu@$PUBLIC_IP:~/`
3. Adapt Phase 1: write a small helper that walks `~/gazettes/*.pdf`, uses `scribe.agent.tools.pdf_to_pages()` for each, and saves per-page PNGs to `data/raw/gazette_pages/` with a manifest JSONL in the same format `label_pages_with_iter3.py` expects
4. Continue from Phase 2 (labeling), pointing `--raw_dir` at `data/raw/gazette_pages/`

---

## Reminders

- **Never touch staging instance** `i-073a0fe419ceb9f49`
- **Rotate keys** after this session
- **Terminate, don't stop** — stopped instances still bill for EBS
- **Bulk labeling is the cost driver** — don't leave the instance idle after Phase 2
- **The ship gate is qualitative** — check tables, seals, bilingual layout, not just chars
