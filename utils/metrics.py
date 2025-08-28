import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted'
    )

    report = classification_report(labels, predictions, target_names=['Class 0', 'Class 1', 'Class 2'])
    print(f"\nClassification Report:\n{report}")

    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }
