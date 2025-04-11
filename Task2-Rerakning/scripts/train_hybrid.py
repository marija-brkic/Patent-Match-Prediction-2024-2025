import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from patent_reranker import (
    ColBERTScorer, 
    CrossEncoderScorer,
    HybridLoss,
    TextExtractorFactory,
    DocumentBatcher
)

# Initialize components
colbert = ColBERTScorer("bert-base-uncased")
cross_encoder = CrossEncoderScorer("cross-encoder/ms-marco-MiniLM-L-12-v2")
loss_fn = HybridLoss(alpha=0.7)
optimizer = torch.optim.AdamW([
    {'params': colbert.parameters()},
    {'params': cross_encoder.parameters()}
], lr=2e-5)

# Training loop
for epoch in range(10):
    for batch in DocumentBatcher(training_data, batch_size=16):
        # 1. Extract text features
        extractor = TextExtractorFactory.create("claim_centric")
        queries = [extractor.extract(q) for q in batch.queries]
        docs = [extractor.extract(d) for d in batch.docs]
        
        # 2. ColBERT scoring
        colbert_scores = colbert.score_batch(queries, docs)
        
        # 3. Cross-encoder refinement
        ce_inputs = [(q, d) for q, d in zip(queries, docs)]
        ce_scores = cross_encoder.score_batch(ce_inputs)
        
        # 4. Calculate hybrid loss
        loss = loss_fn(colbert_scores, ce_scores, batch.labels)
        
        # 5. Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    # Save checkpoint
    torch.save({
        'colbert': colbert.state_dict(),
        'cross_encoder': cross_encoder.state_dict(),
        'optimizer': optimizer.state_dict()
    }, f"hybrid_epoch_{epoch}.pt")