from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import os


def model_fn(model_dir):
    files = os.listdir(model_dir)
    has_bin_files = any(file.endswith('.bin') for file in files)
    model_name = ""    
    
    if has_bin_files:
        model_name = model_dir
    else:
        try:
            with open(f"{model_dir}/model_name.txt", 'r') as file:
                model_name = file.read()
        except FileNotFoundError:
            print("Model name file does not exist.")
            exit(1)

    model_8bit = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="auto", load_in_8bit=True, cache_dir="/tmp/model_cache/")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    return model_8bit, tokenizer


def predict_fn(data, model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    text = data.pop("inputs", data)

    encoded_input = tokenizer(text, return_tensors='pt')
    outputs = model.generate(input_ids=encoded_input['input_ids'].cuda(), **data)
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)