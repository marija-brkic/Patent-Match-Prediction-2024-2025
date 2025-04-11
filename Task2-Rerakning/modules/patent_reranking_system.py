import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from modules.patent_reranker import AdvancedPatentReranker, triplet_loss
from modules.extract_text import extract_text


class PatentRerankingSystem:
    def __init__(self, model_name="intfloat/e5-large-v2", model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AdvancedPatentReranker(model_name=model_name)
        if model_path:
            self._load_model_weights(model_path)
        self.model.to(self.device)
        self.model.eval()

    def _load_model_weights(self, model_path):
        """Handle model loading with compatibility for missing projector weights"""
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict, strict=False)
            print(f"Successfully loaded model weights from {model_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading model weights: {str(e)}")

    def train(self, dataset, epochs=5, lr=2e-5, batch_size=16, save_path="best_model.pth"):
        # Existing training code remains unchanged
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: list(zip(*x)))

        best_loss = float("inf")
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for q, p, n in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                optimizer.zero_grad()
                query, pos, neg = self.model(q, p, n)
                loss = triplet_loss(query, pos, neg)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(self.model.state_dict(), save_path)
                print(f"Saved best model to {save_path}")

    def rerank(self, query_text, doc_texts, batch_size=32, max_length=512):
        # Process query with proper prefix
        query_input = [f"query: {query_text}"]
        
        # Process documents with correct passage prefix
        doc_inputs = [f"passage: {doc}" for doc in doc_texts]

        # Encode query
        with torch.no_grad():
            query_tokens = self.model.tokenizer(
                query_input,
                max_length=max_length,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            query_output = self.model.model(**query_tokens).last_hidden_state
            query_embedding = self.model._mean_pooling(query_output, query_tokens['attention_mask'])
            query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=-1).squeeze(0)

        # Batch process documents
        doc_embeddings = []
        for i in range(0, len(doc_inputs), batch_size):
            batch_docs = doc_inputs[i:i + batch_size]
            doc_tokens = self.model.tokenizer(
                batch_docs,
                max_length=max_length,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            doc_output = self.model.model(**doc_tokens).last_hidden_state
            batch_embeds = self.model._mean_pooling(doc_output, doc_tokens['attention_mask'])
            batch_embeds = torch.nn.functional.normalize(batch_embeds, p=2, dim=-1)
            doc_embeddings.append(batch_embeds)

        doc_embeddings = torch.cat(doc_embeddings, dim=0)

        # Calculate similarity scores
        scores = torch.matmul(doc_embeddings, query_embedding.unsqueeze(-1)).squeeze(-1).tolist()

        # Sort documents by descending score
        sorted_indices = np.argsort(scores)[::-1].tolist()
        return sorted_indices, scores