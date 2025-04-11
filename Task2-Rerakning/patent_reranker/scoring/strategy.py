# scoring/strategy.py
from abc import ABC, abstractmethod
import torch

class IScorer(ABC):
    """Strategy interface for scoring models"""
    @abstractmethod
    def score_batch(self, query: str, docs: list[str]) -> torch.Tensor:
        pass

class ColBERTScorer(IScorer):
    """Implements ColBERT-style late interaction scoring"""
    def __init__(self, model_name="bert-base-uncased"):
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.linear = torch.nn.Linear(768, 128)
        
    def score_batch(self, query, docs):
        # Implementation with late interaction
        return scores

class CrossEncoderScorer(IScorer):
    """Implements cross-attention scoring"""
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-12-v2"):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def score_batch(self, query, docs):
        # Implementation with full cross-attention
        return scores