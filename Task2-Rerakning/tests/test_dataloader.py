import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.data_loader import *

def sanity_check():
    train_ids, test_ids = load_train_test_queries("./data/train_queries.json", "data/test_queries.json")
    query_content = load_content_dict(load_json_data("./data/queries_content_with_features.json"))
    doc_content = load_content_dict(load_json_data("./data/documents_content_with_features.json"))
    pre_ranking = load_pre_ranking("./data/shuffled_pre_ranking.json")

    print("Train queries:", len(train_ids))
    print("Test queries:", len(test_ids))
    print("Query content loaded:", len(query_content))
    print("Doc content loaded:", len(doc_content))
    print("Pre-ranking entries:", len(pre_ranking))

if __name__ == "__main__":
    sanity_check()
