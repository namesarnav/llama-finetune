import torch
from transformers import BitsAndBytesConfig, LlamaTokenizer, AutoTokenizer
from transformers import AutoModelForSequenceClassification


def load_model(model_id: str, hf_token: str, num_labels: int = 3):
    bnb_config = BitsAndBytesConfig(
        # load_in_4bit=True,
        llm_int8_enable_fp32_cpu_offload=True,
        load_in_8bit=True, 
        bnb_8bit_quant_type="fp16",
        bnb_8bit_compute_dtype=torch.float16,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
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
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.pad_token = '[PAD]'
    tokenizer.padding_side = "right"
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    return tokenizer
