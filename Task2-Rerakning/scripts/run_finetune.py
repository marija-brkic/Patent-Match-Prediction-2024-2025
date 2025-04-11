# scripts/run_reranker.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# === run_finetune.py ===
import argparse, json, torch
from torch.utils.data import DataLoader
from modules.patent_reranker import HybridPatentDataset, HybridPatentReranker, train_hybrid_epoch

def load_json_data(path: str) -> dict:
    with open(path, 'r') as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", required=True)
    parser.add_argument("--documents", required=True)
    parser.add_argument("--citations", required=True)
    parser.add_argument("--model", default="intfloat/e5-large-v2")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = (
        torch.device("mps") if torch.backends.mps.is_available() else
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("cpu")
    )
    print(f"Using device: {device}")

    model = HybridPatentReranker("cross-encoder/ms-marco-MiniLM-L-6-v2").to(device)
    model.colbert.gradient_checkpointing_enable()
    try:
        model.cross_encoder.gradient_checkpointing_enable()
    except AttributeError:
        pass

    queries = load_json_data(args.queries)
    documents = load_json_data(args.documents)
    citations = load_json_data(args.citations)

    dataset = HybridPatentDataset(queries, documents, citations, text_type="claim_centric", negatives_per_positive=5)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda x: [i for i in x if i], num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    total_steps = args.epochs * (len(dataset) // args.batch_size)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=2e-4, total_steps=total_steps)

    for epoch in range(args.epochs):
        loss = train_hybrid_epoch(model, dataloader, optimizer, scheduler, device)
        print(f"Epoch {epoch+1} Loss: {loss:.4f}")

    torch.save(model.state_dict(), args.output)
    print(f"Saved hybrid model to {args.output}")

if __name__ == "__main__":
    main()