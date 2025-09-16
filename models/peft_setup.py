from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model


def get_lora_model(model):
    """Prepare a quantized base model and attach LoRA adapters for fine-tuning."""
    # Prepare the frozen base model before introducing trainable LoRA weights.
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.02,
        bias="none",
        task_type="SEQ_CLS",
    )

    return get_peft_model(model, peft_config)
