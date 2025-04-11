# scripts/hybrid_preranker.py
import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch, helpers
from sklearn.preprocessing import MinMaxScaler
import faiss

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def to_patent_dict(items, id_key="FAN"):
    return {str(entry[id_key]): entry["Content"] for entry in items if id_key in entry and "Content" in entry}

def extract_text(content):
    parts = []
    for v in content.values():
        if isinstance(v, str):
            parts.append(v)
        elif isinstance(v, dict):
            parts.append(" ".join(str(x) for x in v.values()))
        elif isinstance(v, list):
            parts.append(" ".join(str(x) for x in v))
        else:
            parts.append(str(v))
    return " ".join(parts).strip()

def normalize(scores):
    if len(scores) == 0:
        return []
    scaler = MinMaxScaler()
    scores_array = np.array(scores).reshape(-1, 1)
    return scaler.fit_transform(scores_array).reshape(-1)

def index_documents(es, index_name, docs):
    if es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)
    actions = [
        {
            "_index": index_name,
            "_id": doc_id,
            "_source": {"content": extract_text(content)}
        }
        for doc_id, content in docs.items()
    ]
    helpers.bulk(es, actions)
    es.indices.refresh(index=index_name)

def es_bm25_search(es, index_name, query_text, top_k=200):
    body = {
        "query": {
            "match": {
                "content": {
                    "query": query_text,
                    "operator": "and"
                }
            }
        },
        "size": top_k
    }
    response = es.search(index=index_name, body=body)
    return [(hit['_id'], hit['_score']) for hit in response['hits']['hits']]

def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index

def run_hybrid_ranking(queries, query_embeddings, doc_embeddings, es, index_name, faiss_index, doc_ids, alpha=0.5, top_k=100):
    results = {}
    print("Running hybrid retrieval...")
    for i, (qid, query_content) in enumerate(tqdm(queries.items())):
        query_text = extract_text(query_content)[:1000]
        query_vector = query_embeddings[i]

        bm25_hits = es_bm25_search(es, index_name, query_text, top_k=200)

        if bm25_hits:
            bm25_doc_ids = [doc_id for doc_id, _ in bm25_hits]
            bm25_scores = [score for _, score in bm25_hits]
            dense_scores = [
                np.dot(query_vector, doc_embeddings[doc_ids.index(doc_id)]) if doc_id in doc_ids else 0.0
                for doc_id in bm25_doc_ids
            ]

            bm25_scores = normalize(bm25_scores)
            dense_scores = normalize(dense_scores)

            hybrid_scores = alpha * np.array(dense_scores) + (1 - alpha) * np.array(bm25_scores)
            ranked_indices = np.argsort(hybrid_scores)[::-1]
            top_docs = [bm25_doc_ids[i] for i in ranked_indices[:top_k]]
        else:
            faiss.normalize_L2(query_vector.reshape(1, -1))
            D, I = faiss_index.search(query_vector.reshape(1, -1), top_k)
            top_docs = [doc_ids[i] for i in I[0]]

        results[qid] = top_docs
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--queries', required=True)
    parser.add_argument('--documents', required=True)
    parser.add_argument('--output', default='outputs/prediction1.json')
    parser.add_argument('--es_index', default='patents_hybrid')
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--model_name', default="sentence-transformers/msmarco-distilbert-base-v3")
    args = parser.parse_args()

    print("[INFO] Loading data...")
    queries_raw = load_json(args.queries)
    documents_raw = load_json(args.documents)
    queries = to_patent_dict(queries_raw, id_key="FAN")
    documents = to_patent_dict(documents_raw, id_key="FAN")
    query_ids = list(queries.keys())
    doc_ids = list(documents.keys())

    print("[INFO] Initializing embedding model...")
    model = SentenceTransformer(args.model_name)

    print("[INFO] Encoding queries and documents...")
    query_texts = [extract_text(queries[qid]) for qid in query_ids]
    doc_texts = [extract_text(documents[did]) for did in doc_ids]
    query_embeddings = model.encode(query_texts, show_progress_bar=True, normalize_embeddings=True)
    doc_embeddings = model.encode(doc_texts, show_progress_bar=True, normalize_embeddings=True)

    print("[INFO] Connecting to Elasticsearch...")
    es = Elasticsearch("http://localhost:9200")
    print(f"[INFO] Indexing {len(documents)} documents into Elasticsearch...")
    index_documents(es, args.es_index, documents)

    print("[INFO] Building FAISS index...")
    faiss_index = build_faiss_index(np.array(doc_embeddings))

    prediction = run_hybrid_ranking(
        queries, np.array(query_embeddings), np.array(doc_embeddings), es,
        args.es_index, faiss_index, doc_ids, alpha=args.alpha
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(prediction, f, indent=2)

    print(f"[INFO] Saved predictions to {args.output}")

if __name__ == '__main__':
    main()