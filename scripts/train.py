"""Launch fine-tuning from a YAML config.

Single GPU:
    python scripts/train.py --config configs/train/lora_lambda_2xa100.yaml

Multi GPU (e.g. 2x A100):
    torchrun --nproc_per_node=2 scripts/train.py --config configs/train/lora_lambda_2xa100.yaml
"""
import argparse
import os

import yaml


def main():
    parser = argparse.ArgumentParser(description="EnclaveScribe fine-tuning")
    parser.add_argument("--config", required=True, help="Path to train config YAML")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        n_gpus = int(os.environ.get("WORLD_SIZE", 1))
        bs = config["training_args"]["per_device_train_batch_size"]
        ga = config["training_args"]["gradient_accumulation_steps"]
        print(f"GPUs: {n_gpus} | per-device batch: {bs} | grad accum: {ga} | effective batch: {bs * ga * n_gpus}")

    from scribe.model.vlm import Qwen2VLModel
    from scribe.train.trainer import train

    model_obj = Qwen2VLModel()
    model_obj.load(
        config["model"]["path"],
        flash_attn=config["model"].get("flash_attn", True),
    )

    if config.get("lora"):
        from scribe.train.lora import apply_lora, build_lora_config
        lora_cfg = build_lora_config(**config["lora"])
        model_obj.model = apply_lora(model_obj.model, lora_cfg)

    train(model_obj.model, model_obj.processor, config)


if __name__ == "__main__":
    main()
