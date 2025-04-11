def train():
    import torch
    import json
    import math
    from torch.utils.data import DataLoader
    from transformers import get_cosine_schedule_with_warmup
    from modules.patent_reranker import HybridPatentDataset, HybridPatentReranker, HybridLoss, train_hybrid_epoch

    def load_json_data(path: str) -> dict:
        with open(path, 'r') as f:
            return json.load(f)

    device = (
        torch.device("mps") if torch.backends.mps.is_available() else
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("cpu")
    )
    print(f"Using device: {device}")

    # Load data
    queries = load_json_data("data/queries.json")
    docs = load_json_data("data/documents.json")
    citations = load_json_data("data/citations.json")

    # Setup
    dataset = HybridPatentDataset(queries, docs, citations, text_type="full", negatives_per_positive=5)
    batch_size = 16
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: [i for i in x if i], num_workers=0)

    model = HybridPatentReranker("intfloat/e5-large-v2").to(device)
    model.colbert.gradient_checkpointing_enable()
    try:
        model.cross_encoder.gradient_checkpointing_enable()
    except AttributeError:
        pass

    optimizer = torch.optim.AdamW([
        {'params': model.colbert.parameters(), 'lr': 1e-6},
        {'params': model.cross_encoder.parameters(), 'lr': 2e-5},
        {'params': model.colbert_proj.parameters()}
    ], weight_decay=0.05)

    total_steps = math.ceil(len(dataset) / batch_size) * 20
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=total_steps)

    for epoch in range(20):
        loss = train_hybrid_epoch(model, dataloader, optimizer, scheduler, device)
        print(f"Epoch {epoch+1} Loss: {loss:.4f}")

    torch.save(model.state_dict(), "hybrid_final_mac.pth")
    print("Model saved as hybrid_final_mac.pth")

if __name__ == "__main__":
    train()