"""

https://huggingface.co/facebook/layerskip-llama3.2-1B

"""

import os
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()
login(token=os.getenv("HF_TOKEN"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from copy import deepcopy

checkpoint = "facebook/layerskip-llama3.2-1B"
early_exit = 4
device = "cuda" if torch.cuda.is_available() else "cpu"
prompt = "typing import List\ndef bucket_sort(A: List):"

model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto", use_safetensors=True, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

generation_config = model.generation_config

weights_memo = {id(w): w for w in model.parameters()}
assistant_model = deepcopy(model, memo=weights_memo) # Clone main model with shared weights
assistant_model.model.layers = assistant_model.model.layers[:early_exit] # Apply early exit
del assistant_model.model.layers[early_exit:]

inputs = tokenizer(prompt, return_tensors="pt").to(device)

outputs = model.generate(**inputs, generation_config=generation_config, assistant_model=assistant_model, max_new_tokens=512)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])