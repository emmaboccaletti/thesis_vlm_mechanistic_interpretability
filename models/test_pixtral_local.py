from mistral_inference.transformer import Transformer

model_path = "/scratch-shared/eboccaletti/models/pixtral_12b_2409"

print("Loading Pixtral...")
model = Transformer.from_folder(model_path)

print("Loaded successfully")