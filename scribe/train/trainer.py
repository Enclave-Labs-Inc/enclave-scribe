from transformers import Trainer, TrainingArguments

from ..data.collator import DocumentCollator
from ..data.dataset import DocumentDataset


def _mask_input_labels(batch: dict) -> dict:
    """Mask padding tokens so loss is not computed on them."""
    labels = batch["input_ids"].clone()
    labels[labels == 0] = -100
    batch["labels"] = labels
    return batch


class ScribeTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        inputs = _mask_input_labels(inputs)
        return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)


def train(model, processor, config: dict) -> None:
    dataset_cfg = config["dataset"]

    train_dataset = DocumentDataset(
        jsonl_path=dataset_cfg["train_jsonl"],
        image_root=dataset_cfg.get("image_root", ""),
        augment=config.get("augment", True),
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

    trainer = ScribeTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model()
