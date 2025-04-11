from transformers import AutoTokenizer, AutoModel
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("intfloat/e5-large-v2")
model = AutoModel.from_pretrained("intfloat/e5-large-v2").eval().to(device)

query = "Smartphone with OLED display"
doc = "A mobile device with an advanced organic screen technology"

inputs = tokenizer([query, doc], return_tensors="pt", padding=True, truncation=True).to(device)
outputs = model(**inputs)
print("Output shape:", outputs.last_hidden_state.shape)
