from transformers import AutoTokenizer, AutoModel
import torch

class CrossEncoderScorer:
    def __init__(self, model_name="intfloat/e5-large-v2", device="cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.device = device

    def score_pairs(self, query, candidates, batch_size=4, max_length=512):
        scores = []
        for i in range(0, len(candidates), batch_size):
            batch = candidates[i:i+batch_size]
            inputs = [f"Re-rank a set of retrieved patents...\nQuery: {query}\nCandidate: {doc}" for doc in batch]
            encodings = self.tokenizer(inputs, truncation=True, padding=True, max_length=max_length, return_tensors='pt').to(self.device)
            with torch.no_grad():
                outputs = self.model(**encodings).last_hidden_state[:, 0]  # CLS token
                scores.extend(outputs[:, 0].cpu().tolist())
        return scores
