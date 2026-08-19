#!/usr/bin/env bash
# Launch an AWS EC2 On-Demand instance for EnclaveScribe training.
#
# Usage:
#   bash scripts/aws/launch.sh
#
# Prerequisites:
#   aws configure  (run once — sets Access Key ID, Secret Key, region)
#   Quota: "Running On-Demand G and VT instances" >= 48 vCPUs (g5.12xlarge uses 48)
#
# Optional env vars (injected into the instance at launch):
#   HF_TOKEN=hf_xxx       — HuggingFace token for gated models
#   WANDB_API_KEY=xxx      — W&B key for live training dashboard (wandb.ai)

set -euo pipefail

# ── Config ───────────────────────────────────────────────────────────────────
INSTANCE_TYPE="${INSTANCE_TYPE:-g5.12xlarge}"   # 4x A10G 24GB, ~$5.67/hr on-demand
REGION="${REGION:-us-east-1}"
AZ="${AZ:-us-east-1c}"
KEY_NAME="${KEY_NAME:-enclave-scribe-key}"
SG_NAME="${SG_NAME:-enclave-scribe-sg}"
VOLUME_SIZE="${VOLUME_SIZE:-500}"              # GB — model + data + checkpoints
# Deep Learning OSS Nvidia Driver AMI — PyTorch 2.7, Ubuntu 22.04, us-east-1
AMI_ID="${AMI_ID:-ami-012ba162b9cd2729c}"

echo "=== EnclaveScribe AWS Launcher ==="
echo "Instance : $INSTANCE_TYPE (on-demand, ~\$5.67/hr)"
echo "Region   : $REGION / $AZ"
echo "AMI      : $AMI_ID"
echo ""

# ── Key pair ─────────────────────────────────────────────────────────────────
if ! aws ec2 describe-key-pairs --key-names "$KEY_NAME" --region "$REGION" &>/dev/null; then
    echo "Creating key pair: $KEY_NAME"
    aws ec2 create-key-pair \
        --key-name "$KEY_NAME" \
        --region "$REGION" \
        --query "KeyMaterial" \
        --output text > "${KEY_NAME}.pem"
    chmod 400 "${KEY_NAME}.pem"
    echo "Key saved → ${KEY_NAME}.pem"
else
    echo "Key pair exists: $KEY_NAME"
    if [ ! -f "${KEY_NAME}.pem" ]; then
        echo "WARNING: ${KEY_NAME}.pem not found locally — SSH will fail."
        echo "         Delete the key pair in AWS console and rerun to regenerate."
    fi
fi

# ── Security group ───────────────────────────────────────────────────────────
SG_ID=$(aws ec2 describe-security-groups \
    --filters "Name=group-name,Values=$SG_NAME" \
    --region "$REGION" \
    --query "SecurityGroups[0].GroupId" \
    --output text 2>/dev/null || echo "None")

if [ "$SG_ID" = "None" ] || [ -z "$SG_ID" ]; then
    echo "Creating security group: $SG_NAME"
    SG_ID=$(aws ec2 create-security-group \
        --group-name "$SG_NAME" \
        --description "EnclaveScribe training" \
        --region "$REGION" \
        --query "GroupId" \
        --output text)
    aws ec2 authorize-security-group-ingress \
        --group-id "$SG_ID" \
        --protocol tcp --port 22 --cidr 0.0.0.0/0 \
        --region "$REGION" > /dev/null
    echo "Security group created: $SG_ID"
else
    echo "Security group exists: $SG_ID"
fi

# ── On-demand instance ───────────────────────────────────────────────────────
echo ""
echo "Launching on-demand instance..."

# Inject secrets as env vars at the top of the UserData script
SECRETS_BLOCK="#!/usr/bin/env bash"
[ -n "${HF_TOKEN:-}" ]       && SECRETS_BLOCK+=$'\n'"export HF_TOKEN=${HF_TOKEN}"
[ -n "${WANDB_API_KEY:-}" ]  && SECRETS_BLOCK+=$'\n'"export WANDB_API_KEY=${WANDB_API_KEY}"
SETUP_SCRIPT=$(tail -n +1 scripts/aws/setup_instance.sh | sed '1s|^#!/usr/bin/env bash||')
COMBINED=$(printf '%s\n%s' "$SECRETS_BLOCK" "$SETUP_SCRIPT")
USERDATA_B64=$(echo "$COMBINED" | base64)

INSTANCE_ID=$(aws ec2 run-instances \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY_NAME" \
    --security-group-ids "$SG_ID" \
    --placement "AvailabilityZone=$AZ" \
    --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":$VOLUME_SIZE,\"VolumeType\":\"gp3\",\"DeleteOnTermination\":true}}]" \
    --user-data "$USERDATA_B64" \
    --count 1 \
    --region "$REGION" \
    --query "Instances[0].InstanceId" \
    --output text)

echo "Instance ID: $INSTANCE_ID"
echo "Waiting for instance to be running..."

aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" --region "$REGION"

PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids "$INSTANCE_ID" \
    --region "$REGION" \
    --query "Reservations[0].Instances[0].PublicIpAddress" \
    --output text)

echo ""
echo "=== Instance Ready ==="
echo "Instance ID : $INSTANCE_ID"
echo "Public IP   : $PUBLIC_IP"
echo ""
echo "SSH command:"
echo "  ssh -i ${KEY_NAME}.pem ubuntu@$PUBLIC_IP"
echo ""
echo "UserData setup runs automatically (~10-15 min: installs deps, starts training)."
echo "Once SSH is available:"
echo "  ssh -i ${KEY_NAME}.pem ubuntu@$PUBLIC_IP"
echo "  tmux attach -t training"
echo ""
echo "Monitor setup progress:"
echo "  tail -f ~/setup.log"
echo ""
echo "IMPORTANT — terminate when done to stop billing:"
echo "  aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region $REGION"

# Save instance info for later
echo "{\"instance_id\": \"$INSTANCE_ID\", \"public_ip\": \"$PUBLIC_IP\"}" \
    > .aws_instance.json
echo ""
echo "Instance info saved → .aws_instance.json"
