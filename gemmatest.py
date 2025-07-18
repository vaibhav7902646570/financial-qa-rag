from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = "./gemma13b_finetuned_model"  # or change this to your actual folder name

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16)

model.eval()  # set model to inference mode
prompt = "Explain the difference between revenue and profit in financial terms."

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=150,         # adjust based on desired length
        temperature=0.7,            # controls randomness
        top_p=0.9,                  # nucleus sampling
        do_sample=True              # True = sampling (more creative), False = deterministic
    )

print(tokenizer.decode(output[0], skip_special_tokens=True))
