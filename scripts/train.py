"""Launch fine-tuning from a YAML config.

Single GPU:
    python scripts/train.py --config configs/train/unsloth_aws.yaml

Multi GPU (torchrun):
    torchrun --nproc_per_node=2 scripts/train.py --config configs/train/unsloth_aws.yaml
"""
import argparse
import os

import yaml


def _unsloth_available() -> bool:
    try:
        import unsloth  # noqa: F401
        return True
    except ImportError:
        return False


def main():
    parser = argparse.ArgumentParser(description="EnclaveScribe fine-tuning")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    use_unsloth = config.get("use_unsloth", True) and _unsloth_available()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    n_gpus = int(os.environ.get("WORLD_SIZE", 1))

    if local_rank == 0:
        bs = config["training_args"]["per_device_train_batch_size"]
        ga = config["training_args"]["gradient_accumulation_steps"]
        backend = "Unsloth" if use_unsloth else "HuggingFace"
        print(f"Backend : {backend}")
        print(f"GPUs    : {n_gpus}")
        print(f"Eff. batch: {bs * ga * n_gpus}")

    if use_unsloth:
        from scribe.train.unsloth_trainer import apply_unsloth_lora, load_unsloth_model, train

        model, tokenizer = load_unsloth_model(
            config["model"]["path"],
            max_seq_length=config.get("max_length", 4096),
        )
        model = apply_unsloth_lora(model, config.get("lora", {}))
        train(model, tokenizer, config)

    else:
        from scribe.model.vlm import Qwen2VLModel
        from scribe.train.lora import apply_lora, build_lora_config
        from scribe.train.trainer import train

        model_obj = Qwen2VLModel()
        model_obj.load(config["model"]["path"], flash_attn=config["model"].get("flash_attn", True))
        if config.get("lora"):
            lora_cfg = build_lora_config(**config["lora"])
            model_obj.model = apply_lora(model_obj.model, lora_cfg)
        train(model_obj.model, model_obj.processor, config)


if __name__ == "__main__":
    main()
