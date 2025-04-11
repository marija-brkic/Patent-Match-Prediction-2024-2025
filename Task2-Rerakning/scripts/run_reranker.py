#!/usr/bin/env python
"""
Script to rerank pre-ranked patent document candidates using a High-MAP Patent Reranker model.
It filters results to only include relevant documents (based on a provided gold mapping).
"""
import os
import sys
import json
import argparse
import logging
from tqdm import tqdm

# Ensure the 'modules' package is in the path (if this script is placed in a subdirectory like 'scripts')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from modules.high_reranker import HighMAPPatentReranker
except Exception as e:
    # If modules are not present, define a placeholder to avoid import errors.
    class HighMAPPatentReranker:
        def __init__(self, bi_encoder_model=None, cross_encoder_model=None):
            logging.warning("HighMAPPatentReranker not found, using dummy model for offline mode.")
        def rerank(self, query_text, doc_texts):
            # Fallback rerank: Sort by descending length of doc_text (dummy logic)
            sorted_indices = sorted(range(len(doc_texts)), key=lambda i: len(doc_texts[i]), reverse=True)
            scores = [len(doc) for doc in doc_texts]
            return sorted_indices, scores

def extract_text(content_dict, text_type="full"):
    """
    Extract text from a patent content dictionary based on the specified text_type.
    """
    if content_dict is None:
        return ""
    if text_type == "TA" or text_type == "title_abstract":
        # Combine title and first abstract paragraph
        title = content_dict.get("title", "")
        abstract = content_dict.get("pa01", "")
        return f"{title} {abstract}".strip()
    elif text_type == "claims":
        # Concatenate all claims (keys starting with 'c-')
        return " ".join([v for k, v in content_dict.items() if k.startswith('c-')])
    elif text_type == "tac1":
        # Title, abstract, and first claim
        title = content_dict.get("title", "")
        abstract = content_dict.get("pa01", "")
        first_claim = ""
        for k, v in content_dict.items():
            if k.startswith('c-'):
                first_claim = v
                break
        return f"{title} {abstract} {first_claim}".strip()
    elif text_type == "description":
        # All description paragraphs (keys starting with 'p')
        paragraphs = [v for k, v in content_dict.items() if k.startswith('p')]
        return " ".join(paragraphs)
    elif text_type == "features":
        # Any field named 'features'
        return content_dict.get("features", "")
    elif text_type == "full":
        # All available text fields: title, abstract, claims, description, etc.
        all_fields = []
        if "title" in content_dict:
            all_fields.append(content_dict["title"])
        if "pa01" in content_dict:
            all_fields.append(content_dict["pa01"])
        # Include everything else (claims, description, etc.)
        for key, value in content_dict.items():
            if key not in ("title", "pa01"):
                all_fields.append(str(value))
        return " ".join(all_fields).strip()
    # Fallback for unknown text_type
    return ""

def load_json_data(file_path):
    """
    Load JSON data from a file path.
    """
    with open(file_path, 'r') as f:
        return json.load(f)

def to_patent_dict(items):
    """
    Convert a list of patent entries into a dictionary mapping ID to Content.
    If the input is already a dictionary, return it as is.
    The ID key is expected to be "FAN" or "doc_id" within each entry.
    """
    if isinstance(items, list):
        if not items:
            return {}
        sample = items[0]
        # Determine which ID key is present (e.g., "FAN" for query patents or "doc_id" for documents)
        if "FAN" in sample:
            return { str(entry["FAN"]): entry["Content"] for entry in items if "FAN" in entry and "Content" in entry }
        if "doc_id" in sample:
            return { str(entry["doc_id"]): entry["Content"] for entry in items if "doc_id" in entry and "Content" in entry }
        raise KeyError("No known ID key ('FAN' or 'doc_id') found in items.")
    # If it's already a dict, assume keys are IDs and values are content dicts
    return items

def rerank_all_queries(queries, documents, pre_ranking, reranker, text_type, top_k=30):
    """
    Rerank candidate documents for each query using the High-MAP reranker.
    Optionally filter to top-k documents based on reranker score.
    """
    reranked_output = {}
    logging.info(f"Loaded {len(pre_ranking)} pre-ranked queries")
    logging.info(f"Loaded {len(queries)} queries and {len(documents)} documents")

    for qid in tqdm(pre_ranking.keys(), desc="Reranking queries"):
        query_content = queries.get(str(qid))
        if not query_content:
            logging.warning(f"Query ID {qid} not found in queries data, skipping.")
            continue

        query_text = extract_text(query_content, text_type)
        if not query_text:
            logging.warning(f"No text found for Query ID {qid}, skipping.")
            continue

        candidate_doc_ids = pre_ranking.get(str(qid), [])
        doc_texts = []
        valid_doc_ids = []

        for doc_id in candidate_doc_ids:
            content = documents.get(str(doc_id))
            if not content:
                continue
            text = extract_text(content, text_type)
            if not text:
                continue
            doc_texts.append(text)
            valid_doc_ids.append(str(doc_id))

        if not doc_texts:
            logging.warning(f"No valid documents for query {qid}, skipping.")
            continue

        sorted_indices, scores = reranker.rerank(query_text, doc_texts)

        ranked_doc_ids = [valid_doc_ids[i] for i in sorted_indices[:top_k]]
        reranked_output[str(qid)] = ranked_doc_ids

    logging.info(f"Finished reranking {len(reranked_output)} queries.")
    return reranked_output


def main():
    parser = argparse.ArgumentParser(description="Rerank patent documents using HighMAPPatentReranker and filter relevant results.")
    parser.add_argument("--queries", type=str, required=True, help="Path to JSON file with query patents content (e.g., queries_content_with_features.json)")
    parser.add_argument("--documents", type=str, required=True, help="Path to JSON file with document patents content (e.g., documents.json)")
    parser.add_argument("--pre_ranking", type=str, required=True, help="Path to JSON file with pre-ranked candidate documents for each query (e.g., shuffled_pre_ranking.json)")
    parser.add_argument("--output", type=str, default="outputs/prediction2.json", help="Path to save the output JSON file with reranked relevant documents")
    parser.add_argument("--text_type", type=str, default="full", help="Type of text content to use for query/doc (default: full)")
    parser.add_argument("--bi_encoder_model", type=str, default="intfloat/e5-large-v2", help="Name or path of the bi-encoder model to use")
    parser.add_argument("--top_k", type=int, default=30, help="How many top reranked docs to keep per query")
    parser.add_argument("--cross_encoder_model", type=str, default="cross-encoder/ms-marco-MiniLM-L-6-v2", help="Name or path of the cross-encoder model to use")
    args = parser.parse_args()

    # Configure logging
    logging.addLevelName(logging.WARNING, "WARN")
    logging.addLevelName(logging.INFO, "INFO")
    logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)

    logging.info("Initializing High-MAP Patent Reranker model...")
    reranker = HighMAPPatentReranker(bi_encoder_model=args.bi_encoder_model,
                                     cross_encoder_model=args.cross_encoder_model)
    logging.info("Loading input data files...")
    queries_data = load_json_data(args.queries)
    documents_data = load_json_data(args.documents)
    pre_ranking = load_json_data(args.pre_ranking)
    # Convert query and document data to dictionary form if needed
    queries = to_patent_dict(queries_data)
    documents = to_patent_dict(documents_data)
    logging.info("Starting reranking process...")
    reranked = rerank_all_queries(
    queries, documents, pre_ranking,
    reranker, args.text_type, top_k=args.top_k
)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(reranked, f, indent=2)
    logging.info(f"Reranked results saved to {args.output}")

if __name__ == "__main__":
    main()
