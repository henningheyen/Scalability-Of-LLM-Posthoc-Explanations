from transformers import pipeline
import numpy as np

classifier = pipeline("zero-shot-classification", model='cross-encoder/nli-deberta-v3-small', use_fast = False)
candidate_labels = candidate_labels=["negative sentiment", "positive sentiment"]

def predict_small(sentences):
  results = classifier(sentences, candidate_labels) 

  # Get the labels from the results
  labels = [pred['labels'] for pred in results]

  # Get the scores from the results
  scores = [pred['scores'] for pred in results]

  # Create NumPy array
  arr = np.zeros((len(sentences), 2))

  # Assign the scores to the corresponding positions in the array
  for i, label in enumerate(labels):

    arr[i,0] = scores[i][label.index(candidate_labels[0])]
    arr[i,1] = scores[i][label.index(candidate_labels[1])]

  return arr

def get_results_small(sentences):
  return classifier(sentences, candidate_labels)

def get_predictions_small(results):

  predicted_labels = [result['labels'][0] for result in results] 
  mapped_labels = [1 if label == 'positive sentiment' else 0 for label in predicted_labels]

  return mapped_labels