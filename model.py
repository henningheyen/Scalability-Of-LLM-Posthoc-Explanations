from transformers import pipeline, GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from torch.utils.data import DataLoader


class ModelZSC:
    """
    A class for zero shot classification using HuggingFace's transformers.
    """    
    def __init__(self, model_name, candidate_labels=None):
        """
        Initialises Zero Shot Classification model. model_name can be an NLI fine tuned model from Huggingface library. 
        """
        self.classifier = pipeline("zero-shot-classification", model=model_name, use_fast=False)
        self.candidate_labels = candidate_labels

    
    def predict(self, sentences, candidate_labels_list=None):
        """
        Takes a list of sentence and a list of candidatte labels (labels for each sentence) and returns numpy array with prediction logits
        """

        if self.candidate_labels is not None:
            candidate_labels_list = [self.candidate_labels]*len(sentences)
        elif candidate_labels_list is None:
            raise ValueError('Provide candidate_labels_list in the predict method or initialise with candidate_labels')

        results = self.get_results(sentences, candidate_labels_list)

        arr = np.zeros((len(sentences), len(candidate_labels_list[0])))

        # Assign the scores to the corresponding positions in the array
        for i, result in enumerate(results):
            current_labels = candidate_labels_list[i]
            for j, label in enumerate(current_labels):
                arr[i, j] = result['scores'][result['labels'].index(label)]
        
        return arr
    
    def get_results(self, sentences, candidate_labels_list=None):
        """
        Returns list of dictionaries with 'seqeunce', 'labels' and 'scores' for each sentence in the list.
        """
        
        if self.candidate_labels is not None:
            candidate_labels_list = [self.candidate_labels]*len(sentences)
        elif candidate_labels_list is None:
            raise ValueError('Provide candidate_labels_list in the predict method or initialise with candidate_labels')

        return [self.classifier(sentence, labels) for sentence, labels in zip(sentences, candidate_labels_list)]


    def get_predictions(self, results, candidate_labels_list=None):
        """
        Takes results list from get_results() method and a list of candidate labels lists
        Returns list of indices corresponding to the predicted lables. Can be used for evaluation
        """
        predicted_labels = [result['labels'][0] for result in results]
        if candidate_labels_list is None:
            mapped_labels = [1 if label == self.candidate_labels[1] else 0 for label in predicted_labels]
        else:
            mapped_labels = [candidate_labels_list[i].index(predicted_labels[i]) for i in range(len(predicted_labels))]
            
        return mapped_labels


class ModelNLI: 
    """
    A class for natural language inference using HuggingFace's transformers.
    """
    def __init__(self, model_name):
        """
        Initialises Natural Language Inference model. model_name can be an NLI fine tuned model from Huggingface library. 
        """
        self.classifier = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast= False)

    def predict(self, sentence_pairs, candidate_labels_list=None):
        """
        Takes a list of premise hypothesis sentence pairs tuples and a list of candidatte labels (labels for each sentence) and returns numpy array with prediction logits
        """   
        features = self.tokenizer(sentence_pairs, padding=True, truncation=True, return_tensors="pt")

        self.classifier.eval()
        with torch.no_grad():
            scores = self.classifier(**features).logits # 0: 'contradiction', 1: 'entailment', 0: 'neutral'
            return torch.nn.functional.softmax(scores, dim=-1).detach().numpy()

    def get_predictions(self, sentence_pairs):
        """
        Returns list of indices corresponding to the predicted lables. Can be used for evaluation
        """

        probs = self.predict(sentence_pairs)
        return [np.argmax(prob) for prob in probs]


