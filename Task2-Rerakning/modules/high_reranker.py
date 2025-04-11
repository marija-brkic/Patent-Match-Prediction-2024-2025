import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from typing import List, Tuple

class HighMAPPatentReranker:
    def __init__(self, 
                 bi_encoder_model: str = "BAAI/bge-large-en-v1.5", 
                 cross_encoder_model: str = "BAAI/bge-reranker-large", 
                 device: str = None, 
                 max_length: int = 512, 
                 cross_max_docs: int = 30):
        """
        Enhanced patent reranker with improved model selection and score alignment.
        """
        # Device setup
        self.device = torch.device(device if device else 
                                  ("cuda" if torch.cuda.is_available() else 
                                   "mps" if torch.backends.mps.is_available() else "cpu"))
        
        # Model initialization
        self.bi_tokenizer = AutoTokenizer.from_pretrained(bi_encoder_model)
        self.bi_model = AutoModel.from_pretrained(bi_encoder_model).to(self.device).eval()
        
        # Cross-encoder setup
        self.cross_encoder_model_name = cross_encoder_model
        if cross_encoder_model:
            self.cross_tokenizer = AutoTokenizer.from_pretrained(cross_encoder_model)
            self.cross_model = AutoModelForSequenceClassification.from_pretrained(cross_encoder_model)
            self.cross_model.to(self.device).eval()
        else:
            self.cross_tokenizer = None
            self.cross_model = None

        self.max_length = max_length
        self.cross_max_docs = cross_max_docs
        self.bi_encoder_model_name = bi_encoder_model.lower()

    def _mean_pool(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Enhanced pooling with mask-aware averaging."""
        expanded_mask = attention_mask.unsqueeze(-1).expand_as(token_embeddings).float()
        return torch.sum(token_embeddings * expanded_mask, 1) / torch.clamp(expanded_mask.sum(1), min=1e-9)

    def _preprocess_text(self, query: str, docs: List[str]) -> Tuple[str, List[str]]:
        """Apply model-specific text preprocessing."""
        if "e5" in self.bi_encoder_model_name:
            return f"query: {query}", [f"passage: {d}" for d in docs]
        elif "bge" in self.bi_encoder_model_name:
            return f"Represent this passage for retrieving relevant passages: {query}", docs
        return query, docs

    def rerank(self, query_text: str, doc_texts: List[str], batch_size: int = 16) -> Tuple[List[int], List[float]]:
        """
        Rerank using paragraph-level bi-encoder and optional cross-encoder scoring.
        """
        if not doc_texts:
            return [], []

        # Preprocess query
        processed_query, _ = self._preprocess_text(query_text, [])
        query_emb = self._embed_texts([processed_query], is_query=True)

        para_texts_per_doc = [self._split_into_paragraphs(doc) for doc in doc_texts]
        flat_para_texts = [p for para_list in para_texts_per_doc for p in para_list]

        if "e5" in self.bi_encoder_model_name:
            flat_para_texts = [f"passage: {t}" for t in flat_para_texts]
        elif "bge" in self.bi_encoder_model_name:
            flat_para_texts = [t for t in flat_para_texts]

        para_embs = self._embed_texts(flat_para_texts, batch_size=batch_size)
        scores = []

        idx = 0
        for para_list in para_texts_per_doc:
            num_paras = len(para_list)
            para_vecs = para_embs[idx:idx+num_paras]
            idx += num_paras
            doc_score = torch.max(torch.matmul(para_vecs, query_emb.T)).item()  # Max similarity per doc
            scores.append(doc_score)

        # Cross-encoder reranking
        cross_indices = self._select_cross_candidates(scores)
        if self.cross_model:
            cross_scores = self._cross_encode(query_text, doc_texts, cross_indices, batch_size)
            scores = self._align_scores(scores, cross_scores, cross_indices)

        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return sorted_indices, [scores[i] for i in sorted_indices]

    def _embed_texts(self, texts: List[str], batch_size: int = 16, is_query: bool = False) -> torch.Tensor:
        """Batch processing for embeddings with device-aware tensors."""
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = self.bi_tokenizer(
                batch, 
                max_length=self.max_length,
                truncation=True, 
                padding=True, 
                return_tensors="pt"
            )
            # Move all input tensors to device first
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.bi_model(**inputs)
                batch_emb = self._mean_pool(
                    outputs.last_hidden_state, 
                    inputs['attention_mask']  # Now on same device
                )
                if "bge" not in self.bi_encoder_model_name:
                    batch_emb = torch.nn.functional.normalize(batch_emb, p=2, dim=1)
                embeddings.append(batch_emb)
        return torch.cat(embeddings, dim=0)

    def _select_cross_candidates(self, scores: List[float]) -> List[int]:
        """Select candidates for cross-encoding with threshold awareness."""
        if not self.cross_model or len(scores) <= self.cross_max_docs:
            return list(range(len(scores)))
        return sorted(sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:self.cross_max_docs])

    def _cross_encode(self, query: str, docs: List[str], indices: List[int], batch_size: int) -> List[float]:
        """Cross-encode with proper device management."""
        pairs = [(query, docs[i]) for i in indices]
        scores = []
        for i in range(0, len(pairs), batch_size):
            inputs = self.cross_tokenizer(
                pairs[i:i+batch_size], 
                max_length=self.max_length,
                truncation=True, 
                padding=True, 
                return_tensors="pt"
            )
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.cross_model(**inputs)
                scores.extend(outputs.logits.squeeze().tolist())
        return scores

    def _align_scores(self, bi_scores: List[float], cross_scores: List[float], indices: List[int]) -> List[float]:
        """Align cross-encoder scores to bi-encoder's scale for consistent ranking."""
        if not cross_scores:
            return bi_scores
            
        # Get score ranges
        bi_subset = [bi_scores[i] for i in indices]
        min_bi, max_bi = min(bi_subset), max(bi_subset)
        min_cross, max_cross = min(cross_scores), max(cross_scores)
        
        # Handle edge cases
        if max_cross == min_cross:
            scaled = [(max_bi + min_bi)/2] * len(cross_scores)
        else:
            scaled = [(s - min_cross)/(max_cross - min_cross)*(max_bi - min_bi) + min_bi 
                     for s in cross_scores]
        
        # Update scores
        aligned_scores = bi_scores.copy()
        for i, s in zip(indices, scaled):
            aligned_scores[i] = s
        return aligned_scores