#!/usr/bin/env bash
# Standard environment bootstrap for EnclaveScribe on AWS Deep Learning AMI.
#
# What this pins and why (all lessons paid for in iter-3 through iter-6):
#   - Use the AMI's /opt/pytorch conda env, not a fresh venv. A fresh venv on
#     the DL AMI silently drops torch to 2.4.1+cu121 (older wheels), while
#     /opt/pytorch has the intended stack. See reports/iter3/env_note.md and
#     reports/iter5/POSTMORTEM.md for the incident.
#   - Pin peft==0.20.0: iter-3/iter-4 adapters use LoraConfig.alora_invocation_tokens,
#     added in peft 0.14; earlier versions cannot load our adapters.
#   - Pin transformers==4.55.4 + accelerate==1.4.0: iter-6 verified this combo
#     loads iter-3/iter-4 adapters and runs training on the AMI's torch.
#   - HF_HUB_DISABLE_XET=1: dodges an xet-acceleration rate-limit (429) we hit
#     during iter-6 OmniDocBench prep.
#
# Uses `set -eo pipefail`; deliberately NOT `-u` because `conda activate`
# references unbound variables internally (iter-6 bootstrap hit this).

set -eo pipefail

CONDA_PROFILE="/opt/conda/etc/profile.d/conda.sh"
if [ ! -f "$CONDA_PROFILE" ]; then
    echo "ERROR: $CONDA_PROFILE not found. This script targets the AWS Deep Learning AMI." >&2
    exit 1
fi

# shellcheck source=/dev/null
source "$CONDA_PROFILE"
conda activate pytorch

pip install --quiet \
    peft==0.20.0 \
    transformers==4.55.4 \
    accelerate==1.4.0

export HF_HUB_DISABLE_XET=1

echo "Environment ready:"
python -c "import torch, transformers, peft, accelerate; \
print(f'  torch        {torch.__version__}'); \
print(f'  transformers {transformers.__version__}'); \
print(f'  peft         {peft.__version__}'); \
print(f'  accelerate   {accelerate.__version__}'); \
print(f'  cuda         {torch.version.cuda} / device {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"cpu\"}')"
echo "  HF_HUB_DISABLE_XET=$HF_HUB_DISABLE_XET"
