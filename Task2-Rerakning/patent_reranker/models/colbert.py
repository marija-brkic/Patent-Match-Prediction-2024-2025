# colbert_reranker.py

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

class ColBERTReranker(torch.nn.Module):
    def __init__(self, model_name="bert-base-uncased", dim=128):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        self.linear = torch.nn.Linear(self.bert.config.hidden_size, dim, bias=False)
        self.dim = dim

    def encode_query(self, text):
        tokens = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            out = self.bert(**tokens)
            rep = self.linear(out.last_hidden_state)  # [B, L, D]
            rep = F.normalize(rep, dim=2)
        return rep, tokens['attention_mask']

    def encode_doc(self, text):
        return self.encode_query(text)  # same encoding logic

    def score(self, query_tokens, query_mask, doc_tokens, doc_mask):
        # Late interaction (max-sim over query tokens)
        q = query_tokens.squeeze(0)  # [Lq, D]
        d = doc_tokens.squeeze(0)    # [Ld, D]
        sim = torch.einsum('qd,kd->qk', q, d)  # [Lq, Ld]
        sim = sim.max(dim=1).values  # [Lq]
        score = sim.sum().item()
        return score

    def rerank(self, query, docs):
        query_vec, query_mask = self.encode_query(query)
        scores = []
        for doc in docs:
            doc_vec, doc_mask = self.encode_doc(doc)
            s = self.score(query_vec, query_mask, doc_vec, doc_mask)
            scores.append(s)
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])
        return ranked, scores