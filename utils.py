import random
from textwrap import dedent
import torch
from transformers import (
    AutoTokenizer
)
import numpy as np


def format_example(input: dict, tokenizer: AutoTokenizer):
    prompt = dedent(
        f"""
    {input["question"]}
        """
    )
    messages = [
        {
            "role": "system",
            "content": "You are a knowledgeable, efficient, and direct AI assistant. Provide concise answers, focusing on the key information needed. Offer suggestions tactfully when appropriate to improve outcomes. Engage in productive collaboration with the user."
        },
        {
            "role": "user",
            "content": prompt
        },
        {
            "role": "assistant",
            "content": ""
        }
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False).rstrip("<|eot_id|>").strip()


def count_tokens(input: str, tokenizer: AutoTokenizer) -> int:
    return len(
        tokenizer(
            input,
            add_special_tokens=True,
            return_attention_mask=False
        )['input_ids']
    )
    
    
# prepare custom dataset(dummy) for speed measurement
def prepare_dataset_for_speed_eval(input: dict, batch_size: int, tokenizer: AutoTokenizer):
    input_batch = []
    for _ in range(batch_size):
        formatted_input = format_example(input, tokenizer)
        input_batch.append(formatted_input)
    
    return tokenizer(input_batch, padding=True, return_tensors="pt")


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    
def quadratic(x, a, b, c):
    return a * x**2 + b * x + c


def linear(x, a, b):
    return a * x + b


def get_profile_penalty(penalty: str):
    if penalty in ["quadratic"]:
        return lambda f, T: np.sum(np.maximum(np.array(f) - np.array(T), 0)**2)
    else:
        raise NotImplementedError("Only quadratic penalty function is supported for now") # will be updated later