#!/usr/bin/env bash
# Standard environment bootstrap for EnclaveScribe on AWS Deep Learning AMI.
#
# What this pins and why (all lessons paid for in iter-3 through iter-7a):
#   - Use the AMI's /opt/pytorch venv (Python 3.12 + torch 2.7.0+cu128 on the
#     PyTorch 2.7 Ubuntu 22.04 DL AMI, ami-012ba162b9cd2729c). Earlier DL AMIs
#     shipped /opt/pytorch as a conda env; the 20260427 image ships it as a
#     plain venv. A fresh `python -m venv` OUTSIDE /opt/pytorch silently picks
#     up torch 2.4.1+cu121 (older wheels), which then breaks liger-kernel and
#     other bits. See reports/iter3/env_note.md, reports/iter5/POSTMORTEM.md,
#     and reports/iter7a/README.md for the incidents.
#   - Pin peft==0.20.0: iter-3/iter-4 adapters use LoraConfig.alora_invocation_tokens,
#     added in peft 0.14; earlier versions cannot load our adapters.
#   - Pin transformers==4.55.4 + accelerate==1.4.0: iter-6 verified this combo
#     loads iter-3/iter-4 adapters and runs training on the AMI's torch. Note:
#     installing `datasets>=2.20` and `trl>=0.12` will pull transformers 5.x
#     as a hard dep; we install those FIRST then re-pin transformers=4.55.4
#     to override (trl 1.x is only used by the Unsloth path we don't take).
#   - HF_HUB_DISABLE_XET=1: dodges an xet-acceleration rate-limit (429) we hit
#     during iter-6 OmniDocBench prep. Set at export, not just pip install.
#   - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True: helps with memory
#     fragmentation on the A10G's 22 GB. Added during iter-7a.
#
# Uses `set -eo pipefail`; deliberately NOT `-u` because activation scripts
# reference unbound variables internally (iter-6 bootstrap hit this).

set -eo pipefail

VENV_ACTIVATE="/opt/pytorch/bin/activate"
CONDA_PROFILE="/opt/conda/etc/profile.d/conda.sh"

if [ -f "$VENV_ACTIVATE" ]; then
    # shellcheck source=/dev/null
    source "$VENV_ACTIVATE"
elif [ -f "$CONDA_PROFILE" ]; then
    # shellcheck source=/dev/null
    source "$CONDA_PROFILE"
    conda activate pytorch
else
    echo "ERROR: neither $VENV_ACTIVATE nor $CONDA_PROFILE found." >&2
    echo "This script targets the AWS Deep Learning AMI (PyTorch flavour)." >&2
    exit 1
fi

# Install training/eval deps first (may transitively upgrade transformers to 5.x),
# then re-pin transformers to 4.55.4 which is what our training loop supports.
pip install --quiet \
    'datasets>=2.20.0' \
    'trl>=0.12.0' \
    liger-kernel \
    peft==0.20.0 \
    accelerate==1.4.0 \
    transformers==4.55.4

export HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Environment ready:"
python -c "import torch, transformers, peft, accelerate; \
print(f'  torch        {torch.__version__}'); \
print(f'  transformers {transformers.__version__}'); \
print(f'  peft         {peft.__version__}'); \
print(f'  accelerate   {accelerate.__version__}'); \
print(f'  cuda         {torch.version.cuda} / device {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"cpu\"}')"
echo "  HF_HUB_DISABLE_XET=$HF_HUB_DISABLE_XET"
echo "  PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
