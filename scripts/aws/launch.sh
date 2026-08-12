#!/usr/bin/env bash
# Launch an AWS EC2 Spot instance for EnclaveScribe training.
#
# Usage:
#   bash scripts/aws/launch.sh
#
# Prerequisites:
#   aws configure  (run once — sets Access Key ID, Secret Key, region)
#   Check g5.12xlarge spot quota: AWS Console → Service Quotas → EC2
#     → "Running On-Demand G and VT instances" → request increase to 48+

set -euo pipefail

# ── Config ───────────────────────────────────────────────────────────────────
INSTANCE_TYPE="${INSTANCE_TYPE:-g5.12xlarge}"   # 4x A10G 24GB, ~$1.70/hr spot
REGION="${REGION:-us-east-1}"
KEY_NAME="${KEY_NAME:-enclave-scribe-key}"
SG_NAME="${SG_NAME:-enclave-scribe-sg}"
SPOT_PRICE="${SPOT_PRICE:-3.00}"               # max bid (on-demand ~$5.67, spot ~$1.70)
VOLUME_SIZE="${VOLUME_SIZE:-500}"              # GB — model + data + checkpoints
# Deep Learning AMI (Ubuntu 22.04) with CUDA 12.1 — update if region changes
AMI_ID="${AMI_ID:-ami-0cf43e1c9a2fe3f27}"     # us-east-1 DL AMI, adjust per region

echo "=== EnclaveScribe AWS Launcher ==="
echo "Instance : $INSTANCE_TYPE"
echo "Region   : $REGION"
echo "Spot max : \$$SPOT_PRICE/hr"
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
        echo "WARNING: ${KEY_NAME}.pem not found locally. SSH will fail."
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

# ── Spot instance ────────────────────────────────────────────────────────────
echo ""
echo "Requesting spot instance..."

LAUNCH_SPEC=$(cat <<EOF
{
  "ImageId": "$AMI_ID",
  "InstanceType": "$INSTANCE_TYPE",
  "KeyName": "$KEY_NAME",
  "SecurityGroupIds": ["$SG_ID"],
  "BlockDeviceMappings": [
    {
      "DeviceName": "/dev/sda1",
      "Ebs": {"VolumeSize": $VOLUME_SIZE, "VolumeType": "gp3", "DeleteOnTermination": true}
    }
  ],
  "UserData": "$(base64 < scripts/aws/setup_instance.sh)"
}
EOF
)

REQUEST_ID=$(aws ec2 request-spot-instances \
    --spot-price "$SPOT_PRICE" \
    --instance-count 1 \
    --type "one-time" \
    --launch-specification "$LAUNCH_SPEC" \
    --region "$REGION" \
    --query "SpotInstanceRequests[0].SpotInstanceRequestId" \
    --output text)

echo "Spot request ID: $REQUEST_ID"
echo "Waiting for instance..."

for i in $(seq 1 40); do
    sleep 15
    INSTANCE_ID=$(aws ec2 describe-spot-instance-requests \
        --spot-instance-request-ids "$REQUEST_ID" \
        --region "$REGION" \
        --query "SpotInstanceRequests[0].InstanceId" \
        --output text 2>/dev/null || echo "")

    STATUS=$(aws ec2 describe-spot-instance-requests \
        --spot-instance-request-ids "$REQUEST_ID" \
        --region "$REGION" \
        --query "SpotInstanceRequests[0].Status.Code" \
        --output text 2>/dev/null || echo "pending")

    echo "  [$i] Status: $STATUS | Instance: ${INSTANCE_ID:-pending}"

    if [ "$STATUS" = "fulfilled" ] && [ -n "$INSTANCE_ID" ] && [ "$INSTANCE_ID" != "None" ]; then
        break
    fi
    if [ "$STATUS" = "capacity-not-available" ] || [ "$STATUS" = "price-too-low" ]; then
        echo "ERROR: Spot not available ($STATUS). Try raising SPOT_PRICE or changing INSTANCE_TYPE."
        exit 1
    fi
done

# Wait for instance to be running
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
echo "Setup runs automatically via UserData (~10 min). Then:"
echo "  ssh -i ${KEY_NAME}.pem ubuntu@$PUBLIC_IP"
echo "  tmux attach -t training"
echo ""
echo "To terminate when done:"
echo "  aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region $REGION"

# Save instance info
echo "{\"instance_id\": \"$INSTANCE_ID\", \"public_ip\": \"$PUBLIC_IP\", \"request_id\": \"$REQUEST_ID\"}" \
    > .aws_instance.json
echo "Instance info saved → .aws_instance.json"
