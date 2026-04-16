from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

torch.cuda.empty_cache()

model_id = "meta-llama/Meta-Llama-3-8B"
# model_id = "meta-llama/Meta-Llama-3-8B"

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    device_map="auto"
)

print("Model loaded successfully!")
