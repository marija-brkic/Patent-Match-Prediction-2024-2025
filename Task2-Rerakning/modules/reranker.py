import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from modules.extract_text import extract_text


class CrossEncoderReranker:
    def __init__(self, model_name="mixedbread-ai/mxbai-embed-large-v1",
                 device=None, batch_size=4, max_length=512):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.batch_size = batch_size
        self.max_length = max_length
        self.sep_token = self.tokenizer.sep_token or "[SEP]"

    def _mxbai_input_format(self, query, document):
        return f"query: {query} {self.sep_token} document: {document}"

    def _mean_pooling(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def _score_docs(self, query_text, doc_texts):
        scores = []
        formatted_inputs = [self._mxbai_input_format(query_text, doc) for doc in doc_texts]

        with torch.no_grad():
            for i in range(0, len(formatted_inputs), self.batch_size):
                batch_texts = formatted_inputs[i:i + self.batch_size]
                batch = self.tokenizer(batch_texts, max_length=self.max_length, padding=True,
                                       truncation=True, return_tensors='pt').to(self.device)
                outputs = self.model(**batch)
                pooled = self._mean_pooling(outputs.last_hidden_state, batch['attention_mask'])
                scores.extend(pooled[:, 0].cpu().tolist())  # Use first dim as relevance proxy
        return scores

    def rerank(self, query_text, doc_texts):
        # Initial scoring
        scores = self._score_docs(query_text, doc_texts)
        initial_ranking = sorted(range(len(scores)), key=lambda i: -scores[i])

        # Pseudo-relevance feedback (top-3 docs)
        feedback_text = " ".join([doc_texts[i] for i in initial_ranking[:3]])
        expanded_query = query_text + " " + feedback_text

        # Final rerank with expanded query
        final_scores = self._score_docs(expanded_query, doc_texts)
        final_ranking = sorted(range(len(final_scores)), key=lambda i: -final_scores[i])
        return final_ranking, final_scores


def rerank_all_queries(queries_dict, docs_dict, pre_ranking, model, text_type="features"):
    reranked = {}
    for qid, doc_ids in tqdm(pre_ranking.items(), desc="Reranking queries"):
        if qid not in queries_dict:
            continue

        query_text = extract_text(queries_dict[qid], text_type)
        doc_texts = [extract_text(docs_dict[doc_id], text_type) for doc_id in doc_ids if doc_id in docs_dict]
        sorted_indices, _ = model.rerank(query_text, doc_texts)
        reranked[qid] = [doc_ids[i] for i in sorted_indices]
    return reranked
