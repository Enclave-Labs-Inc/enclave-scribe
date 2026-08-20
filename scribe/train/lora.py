from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training


def build_lora_config(
    r: int = 64,
    alpha: int = 128,
    dropout: float = 0.05,
    target_modules: list[str] | None = None,
) -> LoraConfig:
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules or [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
    )


def apply_lora(model, config: LoraConfig, qlora: bool = False):
    """Apply LoRA adapters. Set qlora=True when model was loaded in 4/8-bit."""
    if qlora:
        model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model
