from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM
)

def load_model_and_tokenizer(model_name, model_type):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if model_type == "seq2seq":
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    elif model_type == "causal":
        model = AutoModelForCausalLM.from_pretrained(model_name)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return model, tokenizer
