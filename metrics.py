from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
bins = [1,5,10,20,50]


def evaluate(test_labels, predicted_labels):
  return {
    'accuracy': accuracy_score(test_labels, predicted_labels),
    'f1': f1_score(test_labels, predicted_labels, average='weighted'),
    'recall': recall_score(test_labels, predicted_labels, average='weighted'),
    'percision': precision_score(test_labels, predicted_labels, average='weighted'),
  }