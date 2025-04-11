import json


def load_json_data(file_path):
    """Load a JSON file into memory (either a dict or a list)."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_content_dict(json_list):
    """
    From a list of patent entries with 'FAN' and 'Content',
    return a dictionary mapping FAN -> Content.
    """
    return {entry["FAN"]: entry["Content"] for entry in json_list if "FAN" in entry and "Content" in entry}


def load_pre_ranking(file_path):
    """
    Load pre-ranking: mapping from query_id -> list of 30 candidate doc_ids.
    """
    return load_json_data(file_path)


def load_gold_mapping(file_path):
    """
    Load gold labels: mapping from query_id -> list of gold doc_ids.
    """
    return load_json_data(file_path)


def load_query_ids(file_path):
    """
    Load a list of query FANs (used for train or test split).
    """
    return load_json_data(file_path)


def load_train_test_queries(train_path, test_path):
    """
    Load both training and test query IDs.
    """
    return load_query_ids(train_path), load_query_ids(test_path)
