import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

model_path = "/home/eboccaletti/models/qwen2vl7b"

print("Torch:", torch.__version__, flush=True)
print("CUDA available:", torch.cuda.is_available(), flush=True)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0), flush=True)

print("Loading processor...", flush=True)
processor = AutoProcessor.from_pretrained(model_path)

print("Loading model...", flush=True)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"
).eval()

print("Qwen loaded successfully.", flush=True)