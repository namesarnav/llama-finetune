import torch
from transformers import BitsAndBytesConfig, LlamaForSequenceClassification, LlamaTokenizer, AutoTokenizer
from transformers import AutoModelForSequenceClassification


def load_model(model_id: str, hf_token: str, num_labels: int = 3):
    bnb_config = BitsAndBytesConfig(
        # load_in_4bit=True,

        load_in_8bit=True, 
        bnb_4bit_use_double_quant=True,
        bnb_8bit_quant_type="fp16",
        bnb_8bit_compute_dtype=torch.bfloat16,
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
    #tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token # I am pretty sure you can use ['PAD'] as well, because these tokens should be masked out by the attention mask
        tokenizer.padding_side = "right"
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer
