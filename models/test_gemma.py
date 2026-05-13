import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

model_path = "/scratch-shared/eboccaletti/models/gemma3_12b_it"

print("Torch:", torch.__version__, flush=True)
print("CUDA available:", torch.cuda.is_available(), flush=True)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0), flush=True)

print("Loading Gemma...", flush=True)
model = Gemma3ForConditionalGeneration.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.bfloat16
).eval()

processor = AutoProcessor.from_pretrained(model_path)

messages = [
    {"role": "user", "content": [{"type": "text", "text": "Say hello in one short sentence."}]}
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_tensors="pt"
).to(model.device)

print("Running generation...", flush=True)
_ = model.generate(**inputs, max_new_tokens=10)

print("Gemma works.", flush=True)