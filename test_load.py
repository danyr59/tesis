import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel
import sys

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    "./models/profesor",
    torch_dtype="auto",
    device_map="mps",
    trust_remote_code=True,
)
print("Base model loaded successfully!")

print("Loading adapter...")
try:
    model = PeftModel.from_pretrained(base_model, "./models/teacher_adapters")
    print("Adapter loaded successfully!")
except Exception as e:
    print(f"Exception loading adapter: {e}")
    sys.exit(1)
print("Done!")
