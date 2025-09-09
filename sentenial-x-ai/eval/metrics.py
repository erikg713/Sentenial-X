import numpy as np

def latency(times):
    """Compute mean and std-dev of response times (seconds)."""
    return np.mean(times), np.std(times)

def throughput(num_tasks, total_time):
    """Compute number of completed tasks per unit time."""
    return num_tasks / total_time

def accuracy(y_true, y_pred):
    """Fraction of correct predictions (malicious/benign)."""
    return np.mean(np.array(y_true) == np.array(y_pred))

def error_rate(y_true, y_pred):
    """Fraction of incorrect actions."""
    return 1.0 - accuracy(y_true, y_pred)

def precision(y_true, y_pred):
    """Fraction of detected positives that are correct."""
    tp = np.sum((np.array(y_pred) == 1) & (np.array(y_true) == 1))
    fp = np.sum((np.array(y_pred) == 1) & (np.array(y_true) == 0))
    return tp / (tp + fp) if (tp + fp) > 0 else 0

def recall(y_true, y_pred):
    """Fraction of actual positives that are detected."""
    tp = np.sum((np.array(y_pred) == 1) & (np.array(y_true) == 1))
    fn = np.sum((np.array(y_pred) == 0) & (np.array(y_true) == 1))
    return tp / (tp + fn) if (tp + fn) > 0 else 0

def f1_score(y_true, y_pred):
    """Harmonic mean of precision and recall."""
    p, r = precision(y_true, y_pred), recall(y_true, y_pred)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0

def task_completion_rate(completed_tasks, total_tasks):
    """Overall task/incident success rate."""
    return completed_tasks / total_tasks

def resource_utilization(cpu_list, mem_list):
    """Mean CPU and memory per incident."""
    return np.mean(cpu_list), np.mean(mem_list)
  
