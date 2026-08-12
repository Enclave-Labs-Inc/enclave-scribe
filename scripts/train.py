"""Launch fine-tuning from a YAML config."""
import argparse

import yaml


def main():
    parser = argparse.ArgumentParser(description="EnclaveScribe fine-tuning")
    parser.add_argument("--config", required=True, help="Path to train config YAML")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    from scribe.model.vlm import Qwen2VLModel
    from scribe.train.trainer import train

    quantize = config["model"].get("quantize")  # 4 = QLoRA, None = bf16 LoRA
    model_obj = Qwen2VLModel()
    model_obj.load(config["model"]["path"], quantize=quantize)

    if config.get("lora"):
        from scribe.train.lora import apply_lora, build_lora_config
        lora_cfg = build_lora_config(**config["lora"])
        model_obj.model = apply_lora(model_obj.model, lora_cfg, qlora=quantize is not None)

    train(model_obj.model, model_obj.processor, config)


if __name__ == "__main__":
    main()
