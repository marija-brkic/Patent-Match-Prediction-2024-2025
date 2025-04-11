import os
import json
import argparse
import sys

# Import metrics directly from the local file
from modules.metrics import mean_recall_at_k, mean_average_precision, mean_inv_ranking, mean_ranking

def load_json_file(file_path):
    """Load JSON data from a file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description='Evaluate document ranking performance on training data')
    parser.add_argument('--pre_ranking', type=str, default='shuffled_pre_ranking.json',
                        help='Path to pre-ranking JSON file')
    parser.add_argument('--re_ranking', type=str, default='predictions2.json',
                        help='Path to re-ranked JSON file')
    parser.add_argument('--gold', type=str, default='train_gold_mapping.json',
                        help='Path to gold standard mapping JSON file (training only)')
    parser.add_argument('--train_queries', type=str, default='train_queries.json',
                        help='Path to training queries JSON file')
    parser.add_argument('--k_values', type=str, default='3,5,10,20',
                        help='Comma-separated list of k values for Recall@k')
    parser.add_argument('--base_dir', type=str, 
                        default='/bigstorage/younes_internship_2025/CHALLENGE_2025/dataset_files/split_data',
                        help='Base directory for data files')
    args = parser.parse_args()

    # Ensure all paths are relative to base_dir if they're not absolute
    def get_full_path(path):
        if os.path.isabs(path):
            return path
        return os.path.join(args.base_dir, path)

    # Load the training queries
    print("Loading training queries...")
    train_queries = load_json_file(get_full_path(args.train_queries))
    print(f"Loaded {len(train_queries)} training queries")

    # Load the ranking data and gold standard
    print("Loading ranking data and gold standard...")
    pre_ranking = load_json_file(get_full_path(args.pre_ranking))
    re_ranking = load_json_file(get_full_path(args.re_ranking))
    gold_mapping = load_json_file(get_full_path(args.gold))
    
    # Filter to include only training queries
    pre_ranking = {fan: docs for fan, docs in pre_ranking.items() if fan in train_queries}
    re_ranking = {fan: docs for fan, docs in re_ranking.items() if fan in train_queries}  # Fixed this line
    gold_mapping = {fan: docs for fan, docs in gold_mapping.items() if fan in train_queries}
    
    # Parse k values
    k_values = [int(k) for k in args.k_values.split(',')]
    
    # Prepare data for metrics calculation
    query_fans = set(gold_mapping.keys()) & set(pre_ranking.keys()) & set(re_ranking.keys())
    
    if not query_fans:
        print("Error: No common query FANs found across all datasets!")
        return
    
    print(f"Evaluating rankings for {len(query_fans)} training queries...")
    
    # Extract true and predicted labels for both rankings
    true_labels = [gold_mapping[fan] for fan in query_fans]
    pre_ranking_labels = [pre_ranking[fan] for fan in query_fans]
    re_ranking_labels = [re_ranking[fan] for fan in query_fans]
    
    # Calculate metrics for pre-ranking
    print("\nPre-ranking performance (training queries only):")
    for k in k_values:
        recall_at_k = mean_recall_at_k(true_labels, pre_ranking_labels, k=k)
        print(f"  Recall@{k}: {recall_at_k:.4f}")
    
    map_score = mean_average_precision(true_labels, pre_ranking_labels)
    print(f"  MAP: {map_score:.4f}")
    
    inv_rank = mean_inv_ranking(true_labels, pre_ranking_labels)
    print(f"  Mean Inverse Rank: {inv_rank:.4f}")
    
    rank = mean_ranking(true_labels, pre_ranking_labels)
    print(f"  Mean Rank: {rank:.2f}")
    
    # Calculate metrics for re-ranking
    print("\nRe-ranking performance (training queries only):")
    for k in k_values:
        recall_at_k = mean_recall_at_k(true_labels, re_ranking_labels, k=k)
        print(f"  Recall@{k}: {recall_at_k:.4f}")
    
    map_score = mean_average_precision(true_labels, re_ranking_labels)
    print(f"  MAP: {map_score:.4f}")
    
    inv_rank = mean_inv_ranking(true_labels, re_ranking_labels)
    print(f"  Mean Inverse Rank: {inv_rank:.4f}")
    
    rank = mean_ranking(true_labels, re_ranking_labels)
    print(f"  Mean Rank: {rank:.2f}")

if __name__ == "__main__":
    main()
