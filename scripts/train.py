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
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText

        from scribe.train.lora import apply_lora, build_lora_config
        from scribe.train.trainer import train

        model_path = config["model"]["path"]
        flash_attn = config["model"].get("flash_attn", False)
        attn_impl = "flash_attention_2" if flash_attn else "sdpa"

        # For DDP: NO device_map — each rank loads a full copy onto its own GPU.
        # HF Trainer + torchrun handles placement via LOCAL_RANK.
        # min/max_pixels cap image resolution -> caps visual tokens per image
        # (Qwen2.5-VL default max is ~16k*28*28 which can blow up to 10k+ tokens
        # per document scan and mismatch the 4096 max_length).
        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            min_pixels=256 * 28 * 28,      # ~200k px -> ~256 tokens min
            max_pixels=640 * 28 * 28,      # ~500k px -> ~640 tokens max (was 1280 - OOM on 22GB)
        )
        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_impl,
            trust_remote_code=True,
        )

        resume_adapter = config.get("resume_adapter")
        if resume_adapter:
            # Continue training from an existing LoRA adapter (e.g. iter-3).
            # is_trainable=True so PEFT re-enables grad on the adapter weights;
            # without it PeftModel.from_pretrained freezes them.
            from peft import PeftModel
            if local_rank == 0:
                print(f"Resuming LoRA from adapter: {resume_adapter}")
            model = PeftModel.from_pretrained(model, resume_adapter, is_trainable=True)
            model.print_trainable_parameters()
            model.enable_input_require_grads()
        elif config.get("lora"):
            lora_cfg = build_lora_config(**config["lora"])
            model = apply_lora(model, lora_cfg)
            # Required for gradient_checkpointing + LoRA: without this, gradients
            # don't flow through the frozen base model to the trainable adapters
            # and the loss silently stays flat.
            model.enable_input_require_grads()
        train(model, processor, config)


if __name__ == "__main__":
    main()
