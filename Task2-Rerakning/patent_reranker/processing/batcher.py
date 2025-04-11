# processing/batcher.py
from typing import Iterator

class DocumentBatcher:
    """Memory-aware batch iterator"""
    def __init__(self, docs, max_batch_size=32, max_length=512):
        self.docs = docs
        self.max_batch_size = max_batch_size
        self.max_length = max_length
    
    def __iter__(self) -> Iterator[list]:
        batch = []
        current_mem = 0
        for doc in self.docs:
            doc_len = len(doc)
            if current_mem + doc_len > self.max_length * self.max_batch_size:
                yield batch
                batch = []
                current_mem = 0
            batch.append(doc)
            current_mem += doc_len
        if batch:
            yield batch