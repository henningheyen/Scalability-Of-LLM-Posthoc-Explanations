from lime.lime_text import LimeTextExplainer
import numpy as np
import string


class Explainer:
  
  def __init__(self, class_names=None, random_state=42, kernel_width=25, split_expression=lambda x: x.split()):
    """
    Creates an LimeTextExplainer instance with the class names provided
    If not class names are passed Zero Shot Classification setting is assumed
    """
    # For spliting by whitespace and punctuation use split_expression=r'\W+' orinially (we split by whitespace by default)

    self.class_names = class_names
    self.random_state = random_state
    self.kernel_width = kernel_width
    self.split_expression = split_expression
    self.explainer = LimeTextExplainer(class_names=class_names, random_state=random_state, kernel_width=kernel_width, split_expression=split_expression)

  def compute_explanations(self, sentences, model, num_samples=100, num_features=None, task=None, class_names_list=None):
    """
    Computes LIME explanation
    :param sentences: list of sentences (tuple pairs for NLI) to be explained
    :param model: the model used for prediction. Has to implement a predict() method
    :param num_samples: number of perturbed samples to approximate local surrogate model
    :param num_features: number of token to be classified. Per default input token count
    :param task: should be 'NLI' for natural language inference and 'ZSC' for zero shot classification. Can be extended
    :param class_names_list: in case of ZSC requires list of candidate labels accoring to the sentences 
    :return: list of LIME Explanation objects
    """

    explanations = []

    # NLI sentences are pairs of premises and hypotheses
    if task == 'NLI':
      sentences = [sentence[0] + " [SEP] " + sentence[1] for sentence in sentences]
      top_labels=3

    # making pertubations deterministic
    np.random.seed(self.random_state)

    for i, sentence in enumerate(sentences):

      # for e-cos dataset, labels change for each sentence. class_names_list parameter should contain the list aof labels for each sentence
      if class_names_list is not None:
        class_names_list_temp = [class_names_list[i]]*num_samples
        self.explainer = LimeTextExplainer(class_names=class_names_list[i], random_state=self.random_state, kernel_width=self.kernel_width, split_expression=self.split_expression)
        top_labels= len(class_names_list[i])
      else:
         class_names_list_temp = [self.class_names]*num_samples
         top_labels = len(self.class_names)

      # if no specific number is given then set num_feature to the number of tokens
      if num_features is None:
        num_features_temp = len(sentence.split())
      else:
        num_features_temp = num_features
    
      explanation = self.explainer.explain_instance(
         sentence, 
         lambda x: model.predict(x, candidate_labels_list=class_names_list_temp), # [class_names_list_temp[i]]*num_samples will be ignored for ModelNLI objects
         num_samples=num_samples, 
         num_features=num_features_temp, 
         top_labels=top_labels
         )
      explanations.append(explanation)

    return explanations

  def show_lime(self, explanations, show_all_labels=False):
    """
    illustrates LIME explanations. Takes a list of LIME Explanation objects. 
    If show_all_labels=False only displys with respect to thr predicted class
    """
    
    for explanation in explanations:
      if show_all_labels:
        label = None
      else:
        label = [explanation.top_labels[0]] # predicted label
      
      explanation.show_in_notebook(text=True, labels=label)
      print('-'*100)


  def comprehensiveness(self, explanation, sentence, predict, top_k=None, top_percent=None, verbose=True, task='NLI', candidate_labels=None):
    """
    Computes comprehensiveness metric
    :param explanation: LIME explanation to be evaluated
    :param sentence: original input sentence (or sentence pair for NLI)
    :param predict: model predict() method
    :param top_k: calculates comprehensiveness w.r.t. the top k most important tokens
    :param top_percent: calculates comprehensiveness w.r.t top_percent most important tokens
    :param verbose: print computation step for comprhensiveness
    :param task: should be 'NLI' for natrual language inference or 'ZSC' for zero shot classification
    :param candidate_labels: for ZSC setting
    :return: float number representing the metric
    """
      
    if task not in ['NLI', 'ZSC']:
      raise TypeError("Use 'NLI or 'ZSC' for task parameter or extend this method")

    # explanation_tokens is the top_k tokens 
    explanation_tokens = self.get_explanation_tokens(explanation, top_k=top_k, top_percent=top_percent) 

    # Forming new_sentence from explanation_tokens while retaining order from old_sentence
    if task == 'NLI': # Natural Language Inference
      tokens = (sentence[0] + " [SEP] " + sentence[1]).split()
    elif task == 'ZSC': # Zero Shot Classification
      if candidate_labels==None:
        raise TypeError("provide candidate_labels in ZSC setting")
      tokens = sentence.split()

    # Removing the explanation_tokens from the original sentence
    tokens_minus_explanation_tokens = [token for token in tokens if token not in explanation_tokens or token == '[SEP]'] # [SEP] should never be removed for NLI task
    sentence_new = ' '.join(tokens_minus_explanation_tokens)

    # Computing new probability for predicted class
    predicted_class_index = np.argmax(explanation.predict_proba) # predicted class 0: 'contradiction', 1: 'entailment', 2: 'neutral' (NLI)
    if task == 'NLI':
      prediction_new = predict([sentence_new])[0]
      predicted_label_old = self.explainer.class_names[explanation.top_labels[0]]
      predicted_label_new = self.explainer.class_names[np.argmax(prediction_new)]
    elif task == 'ZSC':
      prediction_new = predict([sentence_new], candidate_labels_list=[candidate_labels])[0]
      predicted_label_old = candidate_labels[explanation.top_labels[0]]
      predicted_label_new = candidate_labels[np.argmax(prediction_new)]
      
    probability_new = prediction_new[predicted_class_index]

    # Comparing predictions (with and without explanation_tokens)
    probability_old = max(explanation.predict_proba) #for the predicted class
    comprehensiveness = probability_old - probability_new

    # Print statement
    if verbose:
        print('explanation_tokens: ', explanation_tokens)
        print('sentence_old: ', sentence)
        print('sentence_new: ', sentence_new)    
        print('probability_old: ', probability_old)
        print('predicted_label_old: ', predicted_label_old)
        print('probability_new: ', probability_new)    
        print('predicted_label_new: ', predicted_label_new)
        print('comprehensiveness: ', comprehensiveness, '\n')    

    return comprehensiveness


  def aggregated_metric(self, metric, explanation, sentence, predict, bins=[0.1, 0.3, 0.5], verbose=False, task='NLI', candidate_labels=None):
    """
    calculates aggregrated comprehensiveness based on the provided bins. Similar to ERASER repo: https://github.com/jayded/eraserbenchmark/blob/master/explanation_tokens_benchmark/metrics.py 
    :param explanation: LIME explanation to be evaluated
    :param sentence: original input sentence (or sentence pair for NLI)
    :param predict: model predict() method
    :param top_k: calculates comprehensiveness w.r.t. the top k most important tokens
    :param top_percent: calculates comprehensiveness w.r.t top_percent most important tokens
    :param verbose: print computation step for comprhensiveness
    :param task: should be 'NLI' for natrual language inference or 'ZSC' for zero shot classification
    :param candidate_labels: for ZSC setting
    :return: float number representing the metric
    """

    if metric not in ['comprehensiveness']: # can be extended by more metrics
      raise TypeError("The 'metric' parameter must either 'comprehensiveness'")

    if task not in ['NLI', 'ZSC']:
      raise TypeError("Use 'NLI or 'ZSC' for task parameter or extend this method")

    if task == 'ZSC' and candidate_labels is None:
      raise TypeError("provide 'candidate_labels")

    aggregate = []

    for top_percent in bins:
      comp = self.comprehensiveness(explanation, sentence, predict, top_percent=top_percent, verbose=verbose, task=task, candidate_labels=candidate_labels)
      aggregate.append(comp)    
    return np.mean(aggregate)

  # returns explanation tokens for a single explanation instance
  def get_explanation_tokens(self, explanation, top_k=None, top_percent=None):
    """
    helper function to get the most important tokens
    """
    expl_list = explanation.as_list(label= explanation.top_labels[0])
    expl_list_sorted = sorted(expl_list, key=lambda x: x[1], reverse=True) # sorting in descending order

    if top_k is not None:
        threshold = top_k
    elif top_percent is not None:
        threshold = int(np.ceil(len(expl_list_sorted) * top_percent))
    else:
        # if neither top_k nor top_percent is set then return comprehensiveness for top_k=3 explanation_tokens
        threshold = 3

    explanation_tokens = [token_score_pair[0] for token_score_pair in expl_list_sorted[:threshold]]
    return explanation_tokens

  def get_explanation_list(self, explanations, top_k=None, top_percent=None):
    """
    helper function to get the most important tokens for list of explanations
    """
    return [self.get_explanation_tokens(explanation, top_k=top_k, top_percent=top_percent) for explanation in explanations]


  def compute_macro_iou(self, explanation_tokens, ground_truths, threshold=0.5):
    """
    Computes marco average of intersection over union scores for each pair of explanations (LIME vs. human)
    
    :param explanation_tokens: List of generated explanations (lists of tokens).
    :param ground_truths: List of ground truth explanations (lists of tokens).
    
    :return: mean IOU
    """
    # Check if lists are of the same length
    assert len(explanation_tokens) == len(ground_truths), "Both lists must have the same length."
    
    iou_scores = [self.compute_instance_iou(explanation_tokens, ground_truth) for explanation_tokens, ground_truth in zip(explanation_tokens, ground_truths)]
    return np.mean(iou_scores)


  def compute_instance_iou(self ,explanation_tokens, ground_truth):
    """
    IOU for single instance
    """
    # Remove punctuation and convert to sets
    set_ground_truth = set(word.translate(str.maketrans('', '', string.punctuation)) for word in ground_truth)
    set_explanation_tokens = set(word.translate(str.maketrans('', '', string.punctuation)) for word in explanation_tokens)
    intersection = len(set_explanation_tokens.intersection(set_ground_truth))
    union = len(set_explanation_tokens.union(set_ground_truth))
    return intersection / union if union != 0 else 0


