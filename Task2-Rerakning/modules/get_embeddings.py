import os
import json
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

def load_json(filepath):
    with open(filepath, "r") as f:
        return json.load(f)

def to_patent_dict(items):
    """
    Convert a list of patent entries into a dict with ID -> Content.
    Works for both query and document formats.
    """
    if isinstance(items, dict):
        return items
    elif isinstance(items, list):
        if not items:
            return {}
        if "FAN" in items[0]:
            return {str(entry["FAN"]): entry["Content"] for entry in items if "FAN" in entry and "Content" in entry}
        elif "doc_id" in items[0]:
            return {str(entry["doc_id"]): entry["Content"] for entry in items if "doc_id" in entry and "Content" in entry}
        else:
            raise ValueError("Unknown format for JSON list: can't find 'FAN' or 'doc_id'")
    raise ValueError("Input must be a list or dict.")

def extract_full_text(content_dict):
    """
    Extract concatenated full text from a patent content dictionary.
    """
    fields = []
    if "title" in content_dict:
        fields.append(content_dict["title"])
    if "pa01" in content_dict:
        fields.append(content_dict["pa01"])
    for key, value in content_dict.items():
        if key not in ("title", "pa01"):
            fields.append(str(value))
    return " ".join(fields).strip()

def encode_and_save_texts(model, items_dict, name_prefix):
    """
    Encode dictionary of items and save both embeddings and their IDs.
    """
    ids = list(items_dict.keys())
    texts = [extract_full_text(items_dict[id_]) for id_ in ids]
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    np.save(f"data/{name_prefix}_embeddings.npy", embeddings)
    with open(f"data/{name_prefix}_ids.json", "w") as f:
        json.dump(ids, f)

def main():
    print("[INFO] Loading data...")
    queries_raw = load_json("data/queries_content_with_features.json")
    documents_raw = load_json("data/documents_content_with_features.json")

    queries = to_patent_dict(queries_raw)
    documents = to_patent_dict(documents_raw)

    print("[INFO] Loading SentenceTransformer model...")
    model = SentenceTransformer("intfloat/e5-base-v2")

    print("[INFO] Encoding queries...")
    encode_and_save_texts(model, queries, "query")

    print("[INFO] Encoding documents...")
    encode_and_save_texts(model, documents, "doc")

    print("[INFO] Embeddings saved in `data/` folder.")

if __name__ == "__main__":
    main()
