# utils/loss.py
import torch.nn as nn

class HybridLoss(nn.Module):
    """Composite of contrastive and cross-entropy losses"""
    def __init__(self, alpha=0.7):
        super().__init__()
        self.alpha = alpha
        self.contrastive = nn.TripletMarginLoss()
        self.ce = nn.CrossEntropyLoss()
    
    def forward(self, colbert_scores, ce_scores, labels):
        return self.alpha * self.contrastive(colbert_scores) + \
              (1 - self.alpha) * self.ce(ce_scores, labels)