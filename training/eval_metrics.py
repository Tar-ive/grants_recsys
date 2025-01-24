import numpy as np

def precision_at_k(y_true, y_pred, k=5):
    """Compute Precision@K."""
    top_k_indices = np.argsort(y_pred)[-k:][::-1]
    true_positives = np.sum(y_true[top_k_indices])
    return true_positives / k

def mean_average_precision(y_true, y_pred, k=5):
    """Compute Mean Average Precision (MAP)."""
    average_precisions = []
    for i in range(len(y_true)):
        top_k_indices = np.argsort(y_pred[i])[-k:][::-1]
        precisions = [np.sum(y_true[i][top_k_indices[:j+1]]) / (j+1) for j in range(k)]
        average_precisions.append(np.mean(precisions))
    return np.mean(average_precisions)