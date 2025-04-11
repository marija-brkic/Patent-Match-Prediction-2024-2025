import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
from collections import defaultdict


def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(obj, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2)


def convert_to_finetune_format(
    citation_path,
    citing_content_path,
    cited_content_path,
    out_dir="data"
):
    os.makedirs(out_dir, exist_ok=True)

    print("Loading raw JSON files...")
    citation_data = load_json(citation_path)
    citing_patents = load_json(citing_content_path)
    cited_patents = load_json(cited_content_path)

    print("Indexing content...")
    citing_dict = {entry["Application_Number"]: entry["Content"] for entry in citing_patents}
    cited_dict = {entry["Application_Number"]: entry["Content"] for entry in cited_patents}

    print("Building citation pairs...")
    citations = defaultdict(list)  # citing_id -> list of cited_ids
    for entry in citation_data:
        citing_id, _, cited_id, _, label = entry
        if label in {"X", "Y", "A"}:
            citations[citing_id].append(cited_id)

    print("Saving output JSONs...")
    save_json(citing_dict, os.path.join(out_dir, "queries.json"))
    save_json(cited_dict, os.path.join(out_dir, "documents.json"))
    save_json(citations, os.path.join(out_dir, "citations.json"))
    print("Conversion complete. Files saved to:", out_dir)


if __name__ == "__main__":
    convert_to_finetune_format(
        citation_path="Citation_JSONs/Citation_Train.json",
        citing_content_path="Content_JSONs/Citing_2020_Cleaned_Content_12k/Citing_Train_Test/citing_TRAIN.json",
        cited_content_path="Content_JSONs/Cited_2020_Uncited_2010-2019_Cleaned_Content_22k/CLEANED_CONTENT_DATASET_cited_patents_by_2020_uncited_2010-2019.json",
        out_dir="data"
    )