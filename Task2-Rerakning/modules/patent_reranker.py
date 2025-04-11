# modules/patent_reranker.py

import torch
from torch.utils.data import Dataset, DataLoader  # Added missing import
import random
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import torch.nn.functional as F
from typing import Tuple, Dict, List
from tqdm.auto import tqdm

class HybridPatentDataset(Dataset):
    def __init__(self, queries: Dict, documents: Dict, citations: Dict, 
                 text_type: str = "full", negatives_per_positive: int = 3):
        self.queries = queries
        self.documents = documents
        self.citations = citations
        self.query_ids = list(queries.keys())
        self.doc_ids = list(documents.keys())
        self.negatives_per_positive = negatives_per_positive
        self.text_type = text_type

    def __len__(self) -> int:
        return len(self.query_ids) * self.negatives_per_positive

    def __getitem__(self, idx: int) -> dict:
        query_text, pos_text, neg_text = self._get_triplet(idx)
        if not all([query_text, pos_text, neg_text]):
            return None  # Will be filtered later
        
        return {
            "query": query_text,
            "pos": pos_text,
            "neg": neg_text,
            "pairs": [
                (query_text, pos_text, 1.0),
                (query_text, neg_text, 0.0)
            ]
        }



    def _get_triplet(self, idx: int) -> Tuple[str, str, str]:
        try:
            query_idx = idx % len(self.query_ids)
            query_id = self.query_ids[query_idx]
            pos_ids = self.citations.get(query_id, [])
            
            # If no citations for the query, fallback to a random positive document.
            if not pos_ids:
                pos_id = random.choice(self.doc_ids)
            else:
                pos_id = random.choice(pos_ids)
                # If the chosen citation is not in documents, fallback as well.
                if pos_id not in self.documents:
                    pos_id = random.choice(self.doc_ids)
                    
            # Hard negative mining: choose a negative that isn’t the positive.
            neg_candidates = [d for d in self.doc_ids if d != pos_id]
            if not neg_candidates:
                return ("", "", "")
            neg_id = random.choice(neg_candidates)
            
            return (
                self._extract_text(self.queries.get(query_id, {})),
                self._extract_text(self.documents.get(pos_id, {})),
                self._extract_text(self.documents.get(neg_id, {}))
            )
        except Exception as e:
            # Optionally log the exception e here.
            return ("", "", "")

        
    def _validate_dataset(self):
        """Sanity check dataset contents"""
        sample = self[0]
        print("\nDataset sample check:")
        print(f"Query: {sample[0][:50]}...")
        print(f"Positive: {sample[1][:50]}...")
        print(f"Negative: {sample[2][:50]}...")
        print(f"Pairs: {sample[3]}")
        print(f"Total samples: {len(self)}")
        
    def _extract_text(self, content: Dict) -> str:
        # Enhanced text extraction with claim prioritization
        components = [
            content.get("title", ""),
            content.get("abstract", ""),
            *[v for k, v in content.items() if k.startswith("claim-") and "independent" in v],
            *[v for k, v in content.items() if k.startswith("p")][:3]  # First 3 paragraphs
        ]
        return " ".join(filter(None, components)).strip()

class HybridPatentReranker(torch.nn.Module):
    def __init__(self, model_name: str = "intfloat/e5-large-v2"):
        super().__init__()
        # ColBERT components
        self.colbert_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.colbert = AutoModel.from_pretrained(model_name)
        self.colbert_proj = torch.nn.Linear(self.colbert.config.hidden_size, 128)
        
        # Cross-encoder components
        self.ce_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.cross_encoder = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # Shared components
        self.dropout = torch.nn.Dropout(0.1)
        self.norm = torch.nn.LayerNorm(128)
        
        # Attach the loss function (HybridLoss)
        self.loss_fn = HybridLoss()


    def forward(self, queries: List[str], positives: List[str], negatives: List[str], 
               pairs: List[Tuple[str, str]]) -> Tuple[torch.Tensor, torch.Tensor]:
        # ColBERT encoding
        colbert_outputs = []
        for text in [queries, positives, negatives]:
            inputs = self.colbert_tokenizer(text, padding=True, truncation=True, 
                                          max_length=512, return_tensors="pt").to(self.colbert.device)
            outputs = self.colbert(**inputs).last_hidden_state
            projected = self.norm(self.colbert_proj(outputs))
            colbert_outputs.append(projected)
        
        # Cross-encoder scoring
        ce_scores = []
        for query, doc in pairs:
            inputs = self.ce_tokenizer(query, doc, padding=True, truncation=True,
                                     max_length=512, return_tensors="pt").to(self.cross_encoder.device)
            scores = self.cross_encoder(**inputs).logits
            ce_scores.append(scores)
        
        return colbert_outputs, torch.stack(ce_scores)

class HybridLoss(torch.nn.Module):
    def __init__(self, alpha: float = 0.7, margin: float = 0.3):
        super().__init__()
        self.alpha = alpha
        self.margin = margin
        self.ce_loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, colbert_outputs: list, ce_scores: torch.Tensor, 
               labels: torch.Tensor) -> torch.Tensor:
        # ColBERT triplet loss
        query_emb, pos_emb, neg_emb = colbert_outputs
        pos_sim = F.cosine_similarity(query_emb.mean(1), pos_emb.mean(1))
        neg_sim = F.cosine_similarity(query_emb.mean(1), neg_emb.mean(1))
        
        print(f"Triplet sim - pos: {pos_sim.mean().item():.4f} neg: {neg_sim.mean().item():.4f}")
        print(f"CE scores: {ce_scores.sigmoid().mean().item():.4f}")
        
        triplet_loss = F.relu(neg_sim - pos_sim + self.margin).mean()
        ce_loss = self.ce_loss(ce_scores.squeeze(), labels)
        
        print(f"Triplet loss: {triplet_loss.item():.4f}")
        print(f"CE loss: {ce_loss.item():.4f}")
        
        return self.alpha * triplet_loss + (1 - self.alpha) * ce_loss

def train_hybrid_epoch(model: HybridPatentReranker, dataloader: DataLoader, 
                      optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler._LRScheduler,
                      device: str) -> float:
    model.train()
    total_loss = 0
    batch_counter = 0
    gradient_norms = []
    iterable = tqdm(dataloader) 
    for batch_idx, raw_batch in enumerate(iterable):
        try:
            # ================== DATA VALIDATION ==================
            # Filter invalid samples and check data types
            valid_batch = []
            for sample in raw_batch:
                if sample is None:
                    continue
                # Ensure required keys exist
                if not all(k in sample for k in ("query", "pos", "neg", "pairs")):
                    continue
                # Validate that the texts are strings long enough
                if not (isinstance(sample["query"], str) and len(sample["query"]) > 10 and
                        isinstance(sample["pos"], str) and len(sample["pos"]) > 10 and
                        isinstance(sample["neg"], str) and len(sample["neg"]) > 10):
                    continue
                valid_batch.append(sample)



                
            # Unpack validated batch
            queries = [sample["query"] for sample in valid_batch]
            positives = [sample["pos"] for sample in valid_batch]
            negatives = [sample["neg"] for sample in valid_batch]

            
            # ================== DATA CONVERSION ==================
            # Process pairs and labels
                        
            pair_texts = []
            pair_labels = []
            for sample in valid_batch:
                for pair in sample["pairs"]:
                    if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                        pair_texts.append(pair)
                        # Use the label if present; otherwise default to 0.0
                        pair_labels.append(pair[2] if len(pair) >= 3 else 0.0)
            
            if len(pair_texts) == 0:
                print(f"Skipping batch {batch_idx} because no valid pairs.")
                continue


            
            # ================== FORWARD PASS ==================
            with torch.autocast(device_type=device.type, enabled=True):
                # ColBERT forward
                colbert_outputs = []
                for text_group in [queries, positives, negatives]:
                    inputs = model.colbert_tokenizer(
                        text_group, 
                        padding=True, 
                        truncation=True, 
                        max_length=512, 
                        return_tensors="pt"
                    ).to(device)
                    outputs = model.colbert(**inputs).last_hidden_state
                    projected = model.norm(model.colbert_proj(outputs))
                    colbert_outputs.append(projected)
                
                # Cross-encoder forward
                ce_inputs = model.ce_tokenizer(
                    [p[0] for p in pair_texts],
                    [p[1] for p in pair_texts],
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(device)
                ce_scores = model.cross_encoder(**ce_inputs).logits
                
                # ================== LOSS COMPUTATION ==================
                # Calculate triplet similarity
                query_emb, pos_emb, neg_emb = colbert_outputs
                pos_sim = F.cosine_similarity(query_emb.mean(1), pos_emb.mean(1))
                neg_sim = F.cosine_similarity(query_emb.mean(1), neg_emb.mean(1))
                
                # Hybrid loss calculation
                triplet_loss = F.relu(neg_sim - pos_sim + model.loss_fn.margin).mean()
                labels_tensor = torch.tensor(pair_labels, dtype=torch.float32).to(device)
                ce_loss = model.loss_fn.ce_loss(ce_scores.squeeze(), labels_tensor)

                loss = model.loss_fn.alpha * triplet_loss + (1 - model.loss_fn.alpha) * ce_loss
                
            # ================== BACKPROP ==================
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient monitoring
            total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in model.parameters() if p.grad is not None]), 2)
            gradient_norms.append(total_norm.item())
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            # ================== LOGGING ==================
            if batch_counter % 10 == 0:
                print(f"\nBatch {batch_counter} Diagnostics:")
                print(f"Triplet sim - Pos: {pos_sim.mean().item():.4f} ± {pos_sim.std().item():.4f}")
                print(f"Triplet sim - Neg: {neg_sim.mean().item():.4f} ± {neg_sim.std().item():.4f}")
                print(f"CE scores: {ce_scores.sigmoid().mean().item():.4f}")
                print(f"Triplet loss: {triplet_loss.item():.4f}")
                print(f"CE loss: {ce_loss.item():.4f}")
                print(f"Gradient norm: {total_norm:.4f}")
                
                # Parameter update check
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        update_ratio = torch.mean(torch.abs(param.grad) / (torch.abs(param.data) + 1e-7)).item()
                        print(f"Param {name} update ratio: {update_ratio:.2e}")
            
            total_loss += loss.item()
            batch_counter += 1
            
        except Exception as e:
            print(f"\nError in batch {batch_idx}: {str(e)}")
            print("Sample inputs:")
            if valid_batch:
                sample = valid_batch[0]
                print(f"Query: {sample['query'][:50]}...")
                print(f"Positive: {sample['pos'][:50]}...")
                print(f"Negative: {sample['neg'][:50]}...")
            continue


        # Epoch summary        
    print(f"\nEpoch Summary:")
    if gradient_norms:
        print(f"Avg Gradient Norm: {sum(gradient_norms)/len(gradient_norms):.4f}")
        print(f"Min Gradient Norm: {min(gradient_norms):.4f}")
        print(f"Max Gradient Norm: {max(gradient_norms):.4f}")
    else:
        print("No valid gradients computed this epoch.")
    
    return total_loss / batch_counter if batch_counter > 0 else 0.0
