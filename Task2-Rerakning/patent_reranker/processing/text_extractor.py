# processing/text_extractor.py
from abc import ABC, abstractmethod

class TextExtractionStrategy(ABC):
    """Strategy interface for text extraction"""
    @abstractmethod
    def extract(self, content: dict) -> str:
        pass

class ClaimPriorityExtractor(TextExtractionStrategy):
    def extract(self, content):
        return " ".join([
            content.get("title", ""),
            content.get("abstract", ""),
            *[v for k,v in content.items() if k.startswith('c-') and 'independent' in v],
            *[v for k,v in content.items() if k.startswith('c-') and 'dependent' in v][:3]
        ])

class TextExtractorFactory:
    """Factory for creating extraction strategies"""
    @staticmethod
    def create(style: str) -> TextExtractionStrategy:
        if style == "claim_centric":
            return ClaimPriorityExtractor()
        elif style == "ta":
            return TAExtractor()  # Implement other variants
        else:
            raise ValueError(f"Unknown extraction style: {style}")