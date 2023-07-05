import numpy as np

def get_results(classifier, sentences, candidate_labels):
  return classifier(sentences, candidate_labels)

def predict(classifer, sentences, candidate_labels=["negative sentiment", "positive sentiment"]):
  results = get_results(classifer, sentences, candidate_labels)

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
