from mistral_inference.transformer import Transformer

model_path = "/scratch-shared/eboccaletti/models/pixtral_12b_2409"

print("Loading Pixtral...", flush=True)
model = Transformer.from_folder(model_path)
print("Pixtral loaded successfully.", flush=True)