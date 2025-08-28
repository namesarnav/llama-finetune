import torch
from transformers import BitsAndBytesConfig, LlamaForSequenceClassification, LlamaTokenizer, AutoTokenizer


def load_model(model_id: str, hf_token: str, num_labels: int = 3):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = LlamaForSequenceClassification.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        token=hf_token,
        num_labels=num_labels,
    )
    return model


def load_tokenizer(model_id: str, hf_token: str):
    try:
        tokenizer = LlamaTokenizer.from_pretrained(model_id, token=hf_token)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    return tokenizer
