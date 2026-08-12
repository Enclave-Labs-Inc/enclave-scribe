"""Unsloth-based training — 2x faster, 60% less VRAM, same quality."""
from ..data.dataset import UnslothDocumentDataset


def load_unsloth_model(model_path: str, max_seq_length: int = 4096):
    from unsloth import FastVisionModel

    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        load_in_4bit=False,                    # full bf16 — no quality compromise
        use_gradient_checkpointing="unsloth",  # smarter than HF gradient checkpointing
    )
    return model, tokenizer


def apply_unsloth_lora(model, lora_cfg: dict):
    from unsloth import FastVisionModel

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=lora_cfg.get("r", 128),
        lora_alpha=lora_cfg.get("alpha", 256),
        lora_dropout=lora_cfg.get("dropout", 0),
        bias="none",
        random_state=42,
    )
    model.print_trainable_parameters()
    return model


def train(model, tokenizer, config: dict) -> None:
    from trl import SFTConfig, SFTTrainer
    from unsloth.trainer import UnslothVisionDataCollator

    dataset_cfg = config["dataset"]
    prompt = config.get("prompt", "document parsing.")
    augment = config.get("augment", True)

    train_dataset = UnslothDocumentDataset(
        jsonl_path=dataset_cfg["train_jsonl"],
        image_root=dataset_cfg.get("image_root", ""),
        prompt=prompt,
        augment=augment,
    )
    val_dataset = None
    if dataset_cfg.get("val_jsonl"):
        val_dataset = UnslothDocumentDataset(
            jsonl_path=dataset_cfg["val_jsonl"],
            image_root=dataset_cfg.get("image_root", ""),
            prompt=prompt,
            augment=False,
        )

    train_cfg = config["training_args"]
    args = SFTConfig(
        output_dir=train_cfg["output_dir"],
        num_train_epochs=train_cfg.get("num_train_epochs", 3),
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 2),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 4),
        learning_rate=train_cfg.get("learning_rate", 2e-4),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.05),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        bf16=True,
        fp16=False,
        optim="adamw_8bit",               # Unsloth 8-bit optimizer — no VRAM penalty
        logging_steps=train_cfg.get("logging_steps", 10),
        save_steps=train_cfg.get("save_steps", 300),
        eval_strategy=train_cfg.get("eval_strategy", "steps") if val_dataset else "no",
        eval_steps=train_cfg.get("eval_steps", 300) if val_dataset else None,
        save_total_limit=train_cfg.get("save_total_limit", 5),
        dataloader_num_workers=train_cfg.get("dataloader_num_workers", 4),
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_seq_length=config.get("max_length", 4096),
        report_to="none",
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=args,
    )
    trainer.train()
    trainer.save_model()
