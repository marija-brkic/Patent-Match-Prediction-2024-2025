"""
Evaluation metrics for document ranking.
This file contains implementation of various evaluation metrics 
for assessing the quality of document rankings.
"""
import numpy as np

def recall_at_k(true_items, predicted_items, k=10):
    """
    Calculate recall at k for a single query.
    
    Parameters:
    true_items (list): List of true relevant items
    predicted_items (list): List of predicted items (ranked)
    k (int): Number of top items to consider
    
    Returns:
    float: Recall@k value between 0 and 1
    """
    if not true_items:
        return 0.0  # No relevant items to recall
    
    # Get the top k predicted items
    top_k_items = predicted_items[:k]
    
    # Count the number of true items in the top k predictions
    relevant_in_top_k = sum(1 for item in top_k_items if item in true_items)
    
    # Calculate recall: (relevant items in top k) / (total relevant items)
    return relevant_in_top_k / len(true_items)

def mean_recall_at_k(true_items_list, predicted_items_list, k=10):
    """
    Calculate mean recall at k across multiple queries.
    
    Parameters:
    true_items_list (list of lists): List of true relevant items for each query
    predicted_items_list (list of lists): List of predicted items for each query
    k (int): Number of top items to consider
    
    Returns:
    float: Mean Recall@k value between 0 and 1
    """
    if len(true_items_list) != len(predicted_items_list):
        raise ValueError("Number of true item lists must match number of predicted item lists")
    
    if not true_items_list:
        return 0.0  # No data provided
    
    # Calculate recall@k for each query
    recalls = [recall_at_k(true_items, predicted_items, k) 
               for true_items, predicted_items in zip(true_items_list, predicted_items_list)]
    
    # Return mean recall@k
    return sum(recalls) / len(recalls)

def average_precision(true_items, predicted_items):
    """
    Calculate average precision for a single query.
    
    Parameters:
    true_items (list): List of true relevant items
    predicted_items (list): List of predicted items (ranked)
    
    Returns:
    float: Average precision value between 0 and 1
    """
    if not true_items or not predicted_items:
        return 0.0
    
    # Track number of relevant items seen and running sum of precision values
    relevant_count = 0
    precision_sum = 0.0
    
    # Calculate precision at each position where a relevant item is found
    for i, item in enumerate(predicted_items):
        position = i + 1  # 1-indexed position
        
        if item in true_items:
            relevant_count += 1
            # Precision at this position = relevant items seen / position
            precision_at_position = relevant_count / position
            precision_sum += precision_at_position
    
    # Average precision = sum of precision values / total relevant items
    total_relevant = len(true_items)
    return precision_sum / total_relevant if total_relevant > 0 else 0.0

def mean_average_precision(true_items_list, predicted_items_list):
    """
    Calculate mean average precision (MAP) across multiple queries.
    
    Parameters:
    true_items_list (list of lists): List of true relevant items for each query
    predicted_items_list (list of lists): List of predicted items for each query
    
    Returns:
    float: MAP value between 0 and 1
    """
    if len(true_items_list) != len(predicted_items_list):
        raise ValueError("Number of true item lists must match number of predicted item lists")
    
    if not true_items_list:
        return 0.0  # No data provided
    
    # Calculate average precision for each query
    aps = [average_precision(true_items, predicted_items) 
           for true_items, predicted_items in zip(true_items_list, predicted_items_list)]
    
    # Return mean average precision
    return sum(aps) / len(aps)

def inverse_ranking(true_items, predicted_items):
    """
    Calculate inverse ranking for the first relevant item.
    
    Parameters:
    true_items (list): List of true relevant items
    predicted_items (list): List of predicted items (ranked)
    
    Returns:
    float: Inverse ranking value between 0 and 1
    """
    if not true_items or not predicted_items:
        return 0.0
    
    # Find position of first relevant item (1-indexed)
    for i, item in enumerate(predicted_items):
        if item in true_items:
            rank = i + 1
            return 1.0 / rank  # Inverse ranking
    
    # No relevant items found in predictions
    return 0.0

def mean_inv_ranking(true_items_list, predicted_items_list):
    """
    Calculate mean inverse ranking (MIR) across multiple queries.
    
    Parameters:
    true_items_list (list of lists): List of true relevant items for each query
    predicted_items_list (list of lists): List of predicted items for each query
    
    Returns:
    float: MIR value between 0 and 1
    """
    if len(true_items_list) != len(predicted_items_list):
        raise ValueError("Number of true item lists must match number of predicted item lists")
    
    if not true_items_list:
        return 0.0  # No data provided
    
    # Calculate inverse ranking for each query
    inv_ranks = [inverse_ranking(true_items, predicted_items) 
                 for true_items, predicted_items in zip(true_items_list, predicted_items_list)]
    
    # Return mean inverse ranking
    return sum(inv_ranks) / len(inv_ranks)

def ranking(true_items, predicted_items):
    """
    Calculate the rank of the first relevant item.
    
    Parameters:
    true_items (list): List of true relevant items
    predicted_items (list): List of predicted items (ranked)
    
    Returns:
    float: Rank of the first relevant item (1-indexed)
    """
    if not true_items or not predicted_items:
        return float('inf')  # No relevant items to find
    
    # Find position of first relevant item (1-indexed)
    for i, item in enumerate(predicted_items):
        if item in true_items:
            return i + 1  # Return rank (1-indexed)
    
    # No relevant items found in predictions
    return float('inf')

def mean_ranking(true_items_list, predicted_items_list):
    """
    Calculate mean ranking across multiple queries.
    
    Parameters:
    true_items_list (list of lists): List of true relevant items for each query
    predicted_items_list (list of lists): List of predicted items for each query
    
    Returns:
    float: Mean ranking value (higher is worse)
    """
    if len(true_items_list) != len(predicted_items_list):
        raise ValueError("Number of true item lists must match number of predicted item lists")
    
    if not true_items_list:
        return float('inf')  # No data provided
    
    # Calculate ranking for each query
    ranks = [ranking(true_items, predicted_items) 
             for true_items, predicted_items in zip(true_items_list, predicted_items_list)]
    
    # Filter out 'inf' values for mean calculation
    finite_ranks = [r for r in ranks if r != float('inf')]
    
    # Return mean ranking
    return sum(finite_ranks) / len(finite_ranks) if finite_ranks else float('inf')
