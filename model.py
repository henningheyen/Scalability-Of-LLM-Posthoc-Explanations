from transformers import pipeline
import numpy as np

class SentimentClassifier:
    def __init__(self, model_name, candidate_labels):
        self.classifier = pipeline("zero-shot-classification", model=model_name, use_fast=False)
        self.candidate_labels = candidate_labels

    def predict(self, sentences):
        results = self.get_results(sentences)

        # Get the labels from the results
        labels = [pred['labels'] for pred in results]

        # Get the scores from the results
        scores = [pred['scores'] for pred in results]

        # Create NumPy array
        arr = np.zeros((len(sentences), 2))

        # Assign the scores to the corresponding positions in the array
        for i, label in enumerate(labels):
            arr[i, 0] = scores[i][label.index(self.candidate_labels[0])]
            arr[i, 1] = scores[i][label.index(self.candidate_labels[1])]
        
        return arr

    def get_results(self, sentences):
        return self.classifier(sentences, self.candidate_labels)

    def get_predictions(self, results):
        predicted_labels = [result['labels'][0] for result in results]
        mapped_labels = [1 if label == self.candidate_labels[1] else 0 for label in predicted_labels]

        return mapped_labels
    
