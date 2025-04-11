import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import argparse
from modules.data_loader import load_json_data
from modules.metrics import recall_at_k, mean_average_precision, mean_inv_ranking, mean_ranking

def evaluate(predictions_dict, gold_dict, k_values=[3, 5, 10, 20]):
    print("\nEvaluation Results:")

    # Keep only common query IDs
    query_ids = list(set(predictions_dict) & set(gold_dict))
    if not query_ids:
        print("No common query IDs between predictions and gold labels.")
        return

    predictions = [predictions_dict[qid] for qid in query_ids]
    gold = [gold_dict[qid] for qid in query_ids]

    for k in k_values:
        recall = recall_at_k(gold, predictions, k)
        print(f"  Recall@{k}: {recall:.4f}")

    map_score = mean_average_precision(gold, predictions)
    mir = mean_inv_ranking(gold, predictions)
    mr = mean_ranking(gold, predictions)

    print(f"  MAP: {map_score:.4f}")
    print(f"  Mean Inverse Rank: {mir:.4f}")
    print(f"  Mean Rank: {mr:.2f}\n")

def main():
    parser = argparse.ArgumentParser(description="Evaluate re-ranked patent predictions")
    parser.add_argument("--predictions", type=str, default="outputs/predictions.json")
    parser.add_argument("--gold", type=str, default="data/train_gold_mapping.json")
    parser.add_argument("--k_values", type=str, default="3,5,10,20")
    args = parser.parse_args()

    predictions = load_json_data(args.predictions)
    gold = load_json_data(args.gold)
    k_values = list(map(int, args.k_values.split(",")))

    evaluate(predictions, gold, k_values)

if __name__ == "__main__":
    main()
