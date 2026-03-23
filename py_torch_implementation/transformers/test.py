import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

model_name = "HuggingFaceTB/SmolLM2-135M"

tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

model = PeftModel.from_pretrained(
    base_model,
    "./shakespeare_lora_final",
    adapter_name="shakespeare"
)

model.set_adapter("shakespeare")

model.eval()

prompt = "And now he is trying to"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.8,
        do_sample=True,
        top_p=0.9,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id
    )

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(generated_text)