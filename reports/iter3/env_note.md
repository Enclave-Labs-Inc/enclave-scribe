# iter-3 training environment (best-effort reconstruction, 2026-09-11)

Iter-3 shipped in late August 2026. The instance's `pip freeze` was not
captured at the time. This note reconstructs what is known.

## What's known

Per `reports/iter3/README.md`, iter-3 trained using the AWS Deep Learning
AMI's **pre-baked `/opt/pytorch` conda env**, NOT a fresh `python -m venv`
inside the repo. Explicit quote from that README's incident notes:

> Python 3.11.0rc1 too old — Ubuntu 22.04's `python3.11-venv` installed a
> release candidate predating `sys.get_int_max_str_digits`, breaking
> torch 2.13. Fix: use the AMI's `/opt/pytorch` venv (Python 3.12.10 +
> torch 2.7).

So iter-3's runtime was:

- **Python 3.12.10** (from DL AMI)
- **torch ~2.7** (from DL AMI)
- **transformers**: version not recorded; whatever `pip install -e ".[train,eval]"`
  resolved against the pyproject.toml lower bounds on 2026-08-28
  (`transformers>=4.45.0`) — most likely 4.51-4.54 based on release dates
  around that window.
- **peft ≥ 0.12.0**, **accelerate ≥ 0.33.0**, **trl ≥ 0.12.0**

## What's NOT known

Exact `transformers`, `peft`, `accelerate`, `liger-kernel`, `bitsandbytes`
versions. The training instance was terminated. No pip-freeze artifact was
saved.

## Going-forward rule

**Every future training run publishes `pip freeze` next to its adapter.**
See `outputs/iter5a/SKIPPED.md` and `reports/iter5/POSTMORTEM.md` for the
incident that made this a non-negotiable.

## Practical instruction for reproducing iter-3 inference

Use the DL AMI's `/opt/pytorch` conda env directly rather than creating a
fresh venv. Fresh venvs on this AMI pick up a much older `torch 2.4.1+cu121`
because the pyproject.toml lower bound is `torch>=2.1.0` and pip resolves
whatever wheel is compatible with the installed CUDA — which is not what
the AMI's conda env has cached.

Example on the instance:

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate pytorch
pip install -e ".[eval]"  # installs eval extras against the AMI's torch/transformers
```

## Note on file location

This document lives under `reports/iter3/` (not `outputs/iter3/`) because
`outputs/` is `.gitignore`d — adapters downloaded from S3 shouldn't be
committed, so notes about them are kept in `reports/` alongside the model
card and other docs. Going-forward `pip_freeze.txt` files should follow
the same pattern (`reports/<iter>/pip_freeze.txt`).
