from peft import LoraConfig, TaskType, get_peft_model


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
        target_modules=target_modules or ["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )


def apply_lora(model, config: LoraConfig):
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model
