# pipelines/reranking.py
from dataclasses import dataclass

@dataclass
class RerankingPipeline:
    """Builder for multi-stage reranking process"""
    scorers: list[IScorer]
    batch_size: int = 32
    
    def execute(self, query: str, docs: list[str]):
        current_docs = docs
        for scorer in self.scorers:
            scores = scorer.score_batch(query, current_docs)
            current_docs = self._rerank(current_docs, scores)
        return current_docs
    
    def _rerank(self, docs, scores):
        return [doc for _, doc in sorted(zip(scores, docs), reverse=True)]