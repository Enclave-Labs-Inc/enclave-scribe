from transformers import Trainer, TrainingArguments

from ..data.collator import DocumentCollator
from ..data.dataset import DocumentDataset
from ..data.transforms import apply_train_transforms


def train(model, processor, config: dict) -> None:
    dataset_cfg = config["dataset"]

    train_dataset = DocumentDataset(
        jsonl_path=dataset_cfg["train_jsonl"],
        image_root=dataset_cfg.get("image_root", ""),
        transforms=apply_train_transforms if config.get("augment", True) else None,
    )
    val_dataset = None
    if dataset_cfg.get("val_jsonl"):
        val_dataset = DocumentDataset(
            jsonl_path=dataset_cfg["val_jsonl"],
            image_root=dataset_cfg.get("image_root", ""),
        )

    collator = DocumentCollator(
        processor=processor,
        prompt=config.get("prompt", "document parsing."),
        max_length=config.get("max_length", 4096),
    )

    training_args = TrainingArguments(**config["training_args"])

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model()
