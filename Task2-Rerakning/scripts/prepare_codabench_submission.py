# scripts/prepare_codabench_submission.py

import json

def truncate_and_filter(input_path, output_path, train_queries_path, top_k=30):
    with open(input_path) as f:
        data = json.load(f)

    with open(train_queries_path) as f:
        train_query_ids = set(json.load(f))

    filtered = {
        qid: docs[:top_k]
        for qid, docs in data.items()
        if qid in train_query_ids
    }

    with open(output_path, "w") as f:
        json.dump(filtered, f, indent=2)

    print(f"✅ Wrote {len(filtered)} queries to {output_path}")

if __name__ == "__main__":
    truncate_and_filter(
        input_path="outputs/reranked.json",
        output_path="prediction2.json",
        train_queries_path="data/train_queries.json",
        top_k=30
    )
