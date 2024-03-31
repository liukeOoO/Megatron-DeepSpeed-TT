import torch
import re
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

if __name__ == "__main__":
    token = "" # paste your token from hf here
    model_card = "huggyllama/llama-7b"
    model_card = "huggyllama/llama-13b"
    model_card = "huggyllama/llama-30b"
    model_card = "huggyllama/llama-65b"

    #model_card = "tiiuae/falcon-40b"
    
    tokenizer = AutoTokenizer.from_pretrained(model_card, token=token,)
    model = AutoModelForCausalLM.from_pretrained(model_card, token=token, torch_dtype=torch.float16)
    print(model)

    num_parameters = 0
    for p in model.parameters():
        p_size = p.numel()
        num_parameters += p_size
        print(p_size, p.size())
    num_parameters = num_parameters / 1000.0 / 1000.0 / 1000.0
    print(f"Number of parameters = {num_parameters} Billion")
