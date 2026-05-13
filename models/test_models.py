import torch

# =========================
# QWEN
# =========================
print("===================================")
print("TESTING QWEN")
print("===================================")

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

model_path = "/home/eboccaletti/models/qwen2vl7b"

print("Loading processor...")
processor = AutoProcessor.from_pretrained(model_path)

print("Loading model...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

print("Qwen loaded successfully.\n")


# =========================
# GEMMA
# =========================
print("===================================")
print("TESTING GEMMA")
print("===================================")

from transformers import AutoProcessor, Gemma3ForConditionalGeneration

model_path = "/scratch-shared/eboccaletti/models/gemma3_12b_it"

print("Loading Gemma...")
model = Gemma3ForConditionalGeneration.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.bfloat16
).eval()

processor = AutoProcessor.from_pretrained(model_path)

messages = [
    {"role": "user", "content": [{"type": "text", "text": "What is in this image?"}]}
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_tensors="pt"
).to(model.device)

print("Running small generation...")
_ = model.generate(**inputs, max_new_tokens=10)

print("Gemma works.\n")


# =========================
# PIXTRAL
# =========================
print("===================================")
print("TESTING PIXTRAL")
print("===================================")

from mistral_inference.transformer import Transformer

model_path = "/scratch-shared/eboccaletti/models/pixtral_12b_2409"

print("Loading Pixtral...")
model = Transformer.from_folder(model_path)

print("Pixtral loaded successfully.\n")


print("===================================")
print("ALL MODELS LOADED SUCCESSFULLY")
print("===================================")