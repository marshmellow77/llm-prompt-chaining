from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import os
import subprocess


def model_fn(model_dir):
    output = subprocess.check_output(["df", "-h"])
    print(output.decode('UTF-8'))
    
    files = os.listdir(model_dir)
    print(f"Model directory: {model_dir}")
    print(f"Files: {files}")
    has_bin_files = any(file.endswith('.bin') for file in files)
    print(f"Has .bin files: {has_bin_files}")
    
    model_name = ""
    
    if has_bin_files:
        model_name = model_dir
    else:
        try:
            with open(f"{model_dir}/model_name.txt", 'r') as file:
                model_name = file.read()
        except FileNotFoundError:
            print("File does not exist.")
        
    print(f"Model name: {model_name}")
    print("Model loadining start")

    model_8bit = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="auto", load_in_8bit=True, cache_dir="/tmp/model_cache/")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    output = subprocess.check_output(["df", "-h"])
    print(output.decode('UTF-8'))
    
    print("Model loadining end")

    return model_8bit, tokenizer


def predict_fn(data, model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    text = data.pop("inputs", data)
    print(f"Prompt: {text}")
    encoded_input = tokenizer(text, return_tensors='pt')
    print(f"Data: {data}")
    outputs = model.generate(input_ids=encoded_input['input_ids'].cuda(), **data)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)