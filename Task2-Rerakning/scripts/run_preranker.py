import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch, helpers
import faiss


def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def to_patent_dict(items, id_key="FAN"):
    return {str(entry[id_key]): entry["Content"] for entry in items if id_key in entry and "Content" in entry}

def extract_paragraphs(content):
    paragraphs = []
    for v in content.values():
        if isinstance(v, str):
            paragraphs.append(v)
        elif isinstance(v, dict):
            paragraphs.extend([str(x) for x in v.values()])
        elif isinstance(v, list):
            paragraphs.extend([str(x) for x in v])
    return [p.strip() for p in paragraphs if p.strip()]

def normalize(scores):
    if len(scores) == 0:
        return scores
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
            "_source": {"content": " ".join(extract_paragraphs(content))}
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

def build_faiss_index(doc_embeddings_by_paragraph):
    dim = next(iter(doc_embeddings_by_paragraph.values()))[0].shape[0]
    index = faiss.IndexFlatIP(dim)

    doc_paragraph_ids = []
    all_vectors = []
    for doc_id, para_embeddings in doc_embeddings_by_paragraph.items():
        for emb in para_embeddings:
            doc_paragraph_ids.append(doc_id)
            all_vectors.append(emb)

    vectors = np.vstack(all_vectors)
    faiss.normalize_L2(vectors)
    index.add(vectors)
    return index, doc_paragraph_ids, vectors

def run_paragraph_dense_retrieval(query_embeddings, faiss_index, doc_paragraph_ids, vectors, doc_ids, top_k=100):
    results = {}
    for qid, q_paras in tqdm(query_embeddings.items(), desc="Dense retrieval"):
        doc_score = {}
        for q_emb in q_paras:
            faiss.normalize_L2(q_emb.reshape(1, -1))
            D, I = faiss_index.search(q_emb.reshape(1, -1), 100)
            for idx, score in zip(I[0], D[0]):
                doc_id = doc_paragraph_ids[idx]
                doc_score[doc_id] = doc_score.get(doc_id, 0.0) + float(score)

        ranked = sorted(doc_score.items(), key=lambda x: x[1], reverse=True)
        results[qid] = [doc for doc, _ in ranked[:top_k]]
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--queries', required=True)
    parser.add_argument('--documents', required=True)
    parser.add_argument('--output', default='outputs/prediction1.json')
    parser.add_argument('--es_index', default='patents_hybrid')
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--model_name', default="intfloat/e5-base-v2")
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

    print("[INFO] Encoding paragraphs for queries and documents...")
    query_embeddings = {}
    for qid in tqdm(query_ids, desc="Encoding queries"):
        paras = extract_paragraphs(queries[qid])
        query_embeddings[qid] = model.encode(paras, normalize_embeddings=True)

    doc_embeddings_by_paragraph = {}
    for did in tqdm(doc_ids, desc="Encoding documents"):
        paras = extract_paragraphs(documents[did])
        doc_embeddings_by_paragraph[did] = model.encode(paras, normalize_embeddings=True)

    print("[INFO] Connecting to Elasticsearch...")
    es = Elasticsearch("http://localhost:9200")
    print(f"[INFO] Indexing {len(documents)} documents into Elasticsearch...")
    index_documents(es, args.es_index, documents)

    print("[INFO] Building FAISS index from all document paragraphs...")
    faiss_index, doc_paragraph_ids, all_doc_vectors = build_faiss_index(doc_embeddings_by_paragraph)

    print("[INFO] Running paragraph-level dense retrieval...")
    prediction = run_paragraph_dense_retrieval(query_embeddings, faiss_index, doc_paragraph_ids, all_doc_vectors, doc_ids, top_k=100)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(prediction, f, indent=2)

    print(f"[INFO] Saved predictions to {args.output}")

if __name__ == '__main__':
    main()
