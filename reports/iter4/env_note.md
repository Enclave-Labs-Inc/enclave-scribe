# iter-4 training environment (best-effort reconstruction, 2026-09-11)

Iter-4 shipped in early September 2026. The instance's `pip freeze` was
not captured at the time. This note reconstructs what is known.

## What's known

Iter-4 used the same training scaffolding as iter-3 (same repo, same
`scripts/train.py`, same base model `allenai/olmOCR-2-7B-1025`, same
1× A10G g5.xlarge hardware). It's reasonable to assume iter-4 followed
iter-3's environment convention: **DL AMI's `/opt/pytorch` conda env**,
not a fresh venv. Reasoning:

- Iter-3's README (`reports/iter3/README.md`) explicitly documents this
  as the working setup after the `python3.11-venv` incident.
- Iter-4's runbook (`reports/iter4_runbook.md`) references the same
  provisioning pattern.
- Iter-4 shipped on the same hardware (g5.xlarge, then later work moved
  to g5.4xlarge for RAM headroom during data prep). Fresh venvs on this
  AMI pick up torch 2.4.1+cu121 (old), while `/opt/pytorch` has torch 2.7.

## What today's env-drift attempt taught us

On 2026-09-11 during iter-5 execution, we tried to reproduce iter-4's env
using a fresh `python -m venv` on the DL AMI. Consequences:

- `pip install -e ".[eval]"` resolved to torch 2.4.1+cu121 (whereas
  iter-4 ran on torch ~2.7).
- Under torch 2.4.1, `transformers>=4.55` triggers a memory regression
  vs iter-4's shipped stack — OOM at ~20 GB on 1× A10G 24GB even with
  iter-4's exact hyperparameters.
- Under `transformers==4.49-4.52` (which fixed the memory regression),
  Qwen2.5-VL loading has a `GenerationConfig.to_dict` AttributeError
  and other version-specific bugs.
- `liger-kernel==0.8.2` requires `torch.distributed.tensor.DTensor`
  (added in torch 2.5+), which doesn't exist in torch 2.4.1.

The lesson: fresh venvs on the DL AMI don't match iter-4's shipped env.
See `reports/iter5/POSTMORTEM.md` for the full narrative.

## What's NOT known

Exact `transformers`, `peft`, `accelerate`, `liger-kernel`, `bitsandbytes`
versions. The training instance was terminated. No pip-freeze artifact was
saved. Reasonable guess given late-Aug/early-Sep 2026 timing:
`transformers` ~4.54-4.55, `peft` ~0.13, `accelerate` ~1.0-1.2,
`liger-kernel` ~0.4-0.5.

## Practical instruction for reproducing iter-4 inference

```bash
# on the DL AMI instance:
source /opt/conda/etc/profile.d/conda.sh
conda activate pytorch
pip install -e ".[eval]"  # eval extras against AMI's torch 2.7
```

## Going-forward rule

**Every future training run MUST publish `pip freeze` alongside its
adapter's documentation** as `reports/<iter>/pip_freeze.txt`
(`outputs/` is `.gitignore`d, so committed env records live under
`reports/`). This is a non-negotiable following iter-5's postmortem.
Fresh venvs on the DL AMI are disallowed unless paired with explicit
version pins that match the AMI's conda env.
