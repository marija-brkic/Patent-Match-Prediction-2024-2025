from modules.patent_reranker import PatentRerankingSystem

def main():
    # Initialize system with trained model
    rerank_system = PatentRerankingSystem("final_model.pth")
    
    # Example usage
    query_text = "Solar panel with improved energy efficiency"
    doc_texts = [doc1_text, doc2_text, doc3_text]  # List of document texts
    
    sorted_indices, scores = rerank_system.rerank(query_text, doc_texts)
    print("Reranked documents:", sorted_indices)

if __name__ == "__main__":
    main()