from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model


def get_lora_model(model):
    peft_config = LoraConfig(
        r=16,
        lora_alpha=64,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        lora_dropout=0.02,
        bias="none",
        task_type="SEQ_CLS"
    )
    model = get_peft_model(model, peft_config)
    prepare_model_for_kbit_training(model)
    return model
