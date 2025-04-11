# scripts/run_reranker.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import argparse
from tqdm import tqdm
import torch
from patent_reranker.processing import TextExtractorFactory, DocumentBatcher
from patent_reranker.pipelines import RerankingPipeline
from patent_reranker.scoring import ColBERTScorer, CrossEncoderScorer

def load_json_data(file_path):
    """Loader compatible with existing pipeline"""
    with open(file_path, 'r') as f:
        return json.load(f)

class PatentRerankingSystem:
    """Adapter class integrating new patterns with existing interface"""
    def __init__(self, model_name, model_path=None):
        self.text_extractor = TextExtractorFactory.create("claim_centric")
        self.scorers = self._init_scorers(model_name, model_path)
        self.pipeline = RerankingPipeline(
            scorers=self.scorers,
            batch_size=32
        )
        
    def _init_scorers(self, base_model, checkpoint_path):
        """Factory method for initializing scoring strategies"""
        colbert = ColBERTScorer(base_model)
        cross_encoder = CrossEncoderScorer(base_model)
        
        if checkpoint_path:
            state = torch.load(checkpoint_path)
            colbert.load_state_dict(state['colbert'])
            cross_encoder.load_state_dict(state['cross_encoder'])
            
        return [colbert, cross_encoder]

    def rerank(self, query_text, doc_texts):
        """Execute full reranking pipeline"""
        batcher = DocumentBatcher(doc_texts)
        scores = []
        
        for batch in batcher:
            batch_scores = self.pipeline.execute(query_text, batch)
            scores.extend(batch_scores)
            
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return sorted_indices, scores

def rerank_all_queries(queries, documents, pre_ranking, reranker, text_type):
    """Modified with pattern-based components"""
    reranked_output = {}
    extractor = TextExtractorFactory.create(text_type)

    for qid in tqdm(pre_ranking.keys(), desc="Reranking queries"):
        query_content = queries.get(qid)
        if not query_content:
            continue

        query_text = extractor.extract(query_content)
        doc_ids = pre_ranking[qid]
        
        doc_contents = [documents[did] for did in doc_ids if did in documents]
        doc_texts = [extractor.extract(d) for d in doc_contents]
        
        sorted_indices, _ = reranker.rerank(query_text, doc_texts)
        reranked_output[qid] = [doc_ids[i] for i in sorted_indices]

    return reranked_output

def to_patent_dict(items):
    """Original data loader remains unchanged"""
    if isinstance(items, list):
        sample = items[0]
        for possible_id_key in ["FAN"]:
            if possible_id_key in sample:
                return {x[possible_id_key]: x["Content"] for x in items if possible_id_key in x and "Content" in x}
        raise KeyError("No known ID key found")
    return items

def main():
    """Updated main with pattern integration"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=str, default="data/queries_content_with_features.json")
    parser.add_argument("--documents", type=str, default="data/documents_content_with_features.json")
    parser.add_argument("--pre_ranking", type=str, default="data/shuffled_pre_ranking.json")
    parser.add_argument("--text_type", type=str, default="claim_centric")
    parser.add_argument("--model_ckpt", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--output", type=str, default="outputs/predictions_finetuned.json")
    args = parser.parse_args()

    print("Initializing system...")
    reranker = PatentRerankingSystem(
        model_name=args.model_name,
        model_path=args.model_ckpt
    )

    print("Loading data...")
    queries = to_patent_dict(load_json_data(args.queries))
    documents = to_patent_dict(load_json_data(args.documents))
    pre_ranking = load_json_data(args.pre_ranking)

    print("Starting reranking...")
    reranked = rerank_all_queries(
        queries, documents, pre_ranking, 
        reranker, args.text_type
    )

    print(f"Saving to {args.output}...")
    with open(args.output, "w") as f:
        json.dump(reranked, f, indent=2)

if __name__ == "__main__":
    main()