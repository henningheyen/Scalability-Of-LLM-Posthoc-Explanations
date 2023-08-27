from lime.lime_text import LimeTextExplainer
import numpy as np
import re
import string
from math import sqrt



class Explainer:
  
  def __init__(self, class_names=None, random_state=42, kernel_width=25, split_expression=lambda x: x.split()):
    # split_expression=r'\W+' orinially but we split by whitespace by default

    self.class_names = class_names
    self.random_state = random_state
    self.kernel_width = kernel_width
    self.split_expression = split_expression
    self.explainer = LimeTextExplainer(class_names=class_names, random_state=random_state, kernel_width=kernel_width, split_expression=split_expression)

  def compute_explanations(self, sentences, model, num_samples=100, num_features=None, task=None, class_names_list=None):
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
         lambda x: model.predict(x, candidate_labels_list=class_names_list_temp), # [class_names_list_temp[i]]*num_samples will be ignored for ZeroShotNLI objects
         num_samples=num_samples, 
         num_features=num_features_temp, 
         top_labels=top_labels
         )
      explanations.append(explanation)

    return explanations

  # def compute_explanations(self, sentences, predict, num_samples=100, num_features=None, task=None, class_names_list=None):

  #   explanations = []

  #   # NLI sentences are pairs of premises and hypotheses
  #   if task == 'NLI':
  #     sentences = [sentence[0] + " [SEP] " + sentence[1] for sentence in sentences]
  #     top_labels=3

  #   for i, sentence in enumerate(sentences):

  #     # for e-cos dataset, labels change for each sentence. class_names_list parameter should contain the list aof labels for each sentence
  #     if class_names_list is not None:
  #       self.explainer = LimeTextExplainer(class_names=class_names_list[i], random_state=self.random_state)
  #       top_labels= len(class_names_list[i])

  #     # if no specific number is given then set num_feature to the number of tokens
  #     if num_features is None:
  #       num_features_temp = len(sentence.split())
  #     else:
  #       num_features_temp = num_features

  #     explanation = self.explainer.explain_instance(sentence, predict, num_samples=num_samples, num_features=num_features_temp, top_labels=top_labels)
  #     explanations.append(explanation)

  #   return explanations

  def show_lime(self, explanations, show_all_labels=False):
    
    for explanation in explanations:
      if show_all_labels:
        label = None
      else:
        label = [explanation.top_labels[0]]
      
      explanation.show_in_notebook(text=True, labels=label)
      print('-'*100)


  def comprehensiveness(self, explanation, sentence, predict, top_k=None, top_percent=None, verbose=True, task='NLI', candidate_labels=None):
      
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


  def sufficiency(self, explanation, sentence, predict, top_k=None, top_percent=None, verbose=True, task='NLI', candidate_labels=None):

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
    
    sentence_new = ' '.join(token for token in tokens if token in explanation_tokens or token == "[SEP]")

    # Computing new probability for predicted class
    predicted_class_index = np.argmax(explanation.predict_proba) # predicted class 0: 'contradiction', 1: 'entailment', 2: 'neutral'
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

    sufficiency = probability_old - probability_new

    if verbose:
        print('explanation_tokens: ', explanation_tokens)
        print('sentence_old: ', sentence)    
        print('sentence_new: ', sentence_new)    
        print('probability_old: ', probability_old)
        print('predicted_label_old: ', predicted_label_old)
        print('probability_new: ', probability_new)    
        print('predicted_label_new: ', predicted_label_new)
        print('sufficiency: ', sufficiency, '\n')

    return sufficiency

  def aggregated_metric(self, metric, explanation, sentence, predict, bins=[0.1, 0.3, 0.5], verbose=False, task='NLI', candidate_labels=None):

    if metric not in ['comprehensiveness', 'sufficiency']:
      raise TypeError("The 'metric' parameter must either 'comprehensiveness' or 'sufficiency'")

    if task not in ['NLI', 'ZSC']:
      raise TypeError("Use 'NLI or 'ZSC' for task parameter or extend this method")

    if task == 'ZSC' and candidate_labels is None:
      raise TypeError("provide 'candidate_labels")

    aggregate = []

    for top_percent in bins:
      if metric == 'comprehensiveness':
        comp = self.comprehensiveness(explanation, sentence, predict, top_percent=top_percent, verbose=verbose, task=task, candidate_labels=candidate_labels)
        aggregate.append(comp)    
      else:
        suff = self.sufficiency(explanation, sentence, predict, top_percent=top_percent, verbose=verbose, task=task, candidate_labels=candidate_labels)
        aggregate.append(suff)    
    return np.mean(aggregate)

  # returns explanation tokens for a single explanation instance
  def get_explanation_tokens(self, explanation, top_k=None, top_percent=None):
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
    return [self.get_explanation_tokens(explanation, top_k=top_k, top_percent=top_percent) for explanation in explanations]
  

  # If LIME was initialised with split_expression=r'\W+' instead of lambda x: x.split() then use this method to remove tokens
  def remove_explanation_tokens_from_sentence(sentence, explanation_tokens): 
    # Split the sentence into words using regex
    words = re.findall(r'\w+|[.,!?;]|\'\w+', old)

    # Removing the rationale from the original sentence while preserving punctuation and apostrophes
    sentence_new = []

    for word in words:
        if re.match(r'\w+', word):
            if word not in rationale:
                sentence_new.append(word)
        else:
            sentence_new.append(word)


    # Join the words to form the new sentence
    sentence_new = ' '.join(sentence_new).replace(' ,', ',').replace(' .', '.').replace(' ?', '?').replace(' !', '!').replace(' ;', ';').replace(" '", "'").replace(' :', ':')

    print(sentence_new)


  # def lime_tokenize(self, sentence, split_expression=r'\W+'):
  #   """
  #   Tokenizes the sentence based on the provided split_expression same as LIME documentation (splits on non-word characters).
  #   """
  #   # Using the split_expression as a non-capturing group
  #   splitter = re.compile(r'(%s)|$' % split_expression)
  #   return [s for s in splitter.split(sentence) if s]
  
  # def get_explanation_list(self, explanations, top_percent):
    
  #   explanation_list = []
    
  #   for explanation in explanations:
  #       expl_list = [explanation.as_list(label= explanation.top_labels[0])[i][0].lower() for i in range(len(explanation.as_list()))]
  #       threshold = int(np.ceil(len(expl_list) * top_percent))
  #       explanation_list.append(expl_list[:threshold])
    
  #   return explanation_list
  
  def format_explanation_true_list(self, explanation_true_list):

    # spliting words with apostrophe ('wasn't' -> 'wasn', 't')
    explanation_true_list = [[word for item in explanation for word in item.replace("'", " ").split()] for explanation in explanation_true_list]

    # Now split the words by spaces and remove punctuation ('kissing?' -> 'kissing')
    translator = str.maketrans('', '', string.punctuation)
        
    # Use a list comprehension to remove punctuation from each word in each list.
    explanation_true_list = [[word.translate(translator) for word in word_list] for word_list in explanation_true_list]

    return explanation_true_list
  
  # def compute_token_f1(self, explanation_true, explanation_pred):
  #   true_set = set(explanation_true)
  #   pred_set = set(explanation_pred)

  #   # Calculate precision and recall
  #   precision = len(true_set & pred_set) / len(pred_set) if pred_set else 0
  #   recall = len(true_set & pred_set) / len(true_set) if true_set else 0

  #   # Calculate F1 score
  #   f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

  #   return f1


  def compute_macro_iou(self, explanation_tokenss, ground_truths, threshold=0.5):
      """
      Computes IoU values for each pair of explanations from two lists and checks if they exceed a threshold.
      
      Args:
      - explanation_tokenss: List of explanations (lists of tokens).
      - ground_truths: List of explanations (lists of tokens).
      - threshold: IoU threshold to consider a match.
      
      Returns:
      - average of List of fractions indiction the IOU for each instance
      """
      # Check if lists are of the same length
      if len(explanation_tokenss) != len(ground_truths):
          raise ValueError("Both lists should be of the same length.")
      
      iou_scores = [self.compute_instance_iou(explanation_tokens, ground_truth) for explanation_tokens, ground_truth in zip(explanation_tokenss, ground_truths)]
      return np.mean(iou_scores)


  def compute_instance_iou(self ,explanation_tokens, ground_truth):
    """Helper function to compute IoU for two explanations."""
    set_explanation_tokens = set(explanation_tokens)
    set_ground_truth = set(ground_truth)
    intersection = len(set_explanation_tokens.intersection(set_ground_truth))
    union = len(set_explanation_tokens.union(set_ground_truth))
    return intersection / union if union != 0 else 0


  # similar to ERASER repo: https://github.com/jayded/eraserbenchmark/blob/master/explanation_tokens_benchmark/metrics.py 
  def compute_instance_f1(self, explanation_tokens, ground_truth):
      """
      Computes the instance F1 score for two given lists of tokens.
      """
      set_explanation_tokens = set(explanation_tokens)
      set_ground_truth = set(ground_truth)
      
      tp = len(set_explanation_tokens.intersection(set_ground_truth))
      fp = len(set_explanation_tokens.difference(set_ground_truth))
      fn = len(set_ground_truth.difference(set_explanation_tokens))
      
      precision = tp / (tp + fp) if (tp + fp) != 0 else 0
      recall = tp / (tp + fn) if (tp + fn) != 0 else 0
      f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0
      
      return f1

  def compute_macro_f1(self, explanation_tokenss, ground_truths):
      """
      Computes the macro F1 score for two given lists of lists of tokens.
      """
      assert len(explanation_tokenss) == len(ground_truths), "Both lists must have the same length."
      
      f1_scores = [self.compute_instance_f1(r, gt) for r, gt in zip(explanation_tokenss, ground_truths)]
      
      macro_f1 = sum(f1_scores) / len(f1_scores)
      
      return macro_f1




  #a = ['hello', 'world', 'I', 'am', 'bored']
  #b = ['hello', 'you', 'I', 'am', 'bored', 'can']

  #print(cosine_similarity(a, b))







  #Old version
  # def compute_explanations2(self, sentences, predict, num_samples=100, num_features=None, task=None, class_names_list=None):

  #   explanations = []

  #   # NLI sentences are pairs of premises and hypotheses
  #   if task == 'NLI':
  #     sentences = [sentence[0] + " [SEP] " + sentence[1] for sentence in sentences]
  #     top_labels=3

  #   for i, sentence in enumerate(sentences):

  #     # for e-cos dataset, labels change for each sentence. class_names_list parameter should contain the list aof labels for each sentence
  #     if class_names_list is not None:
  #       self.explainer = LimeTextExplainer(class_names=class_names_list[i], random_state=self.random_state)
  #       top_labels= len(class_names_list[i])

  #     # if no specific number is given then set num_feature to the number of tokens
  #     if num_features is None:
  #       num_features_temp = len(sentence.split())
  #     else:
  #       num_features_temp = num_features

  #     explanation = self.explainer.explain_instance(sentence, predict, num_samples=num_samples, num_features=num_features_temp, top_labels=top_labels)
  #     explanations.append(explanation)

  #   return explanations

# class Explainer2:

#   def __init__(self, class_names=None, random_state=42):
#     self.class_names = class_names
#     self.random_state = random_state
#     self.explainer = LimeTextExplainer(class_names=class_names, random_state=random_state)

#   def compute_explanations(self, sentences, predict, num_samples=100, num_features=None, task=None, class_names_list=None):

#     explanations = []

#     # NLI sentences are pairs of premises and hypotheses
#     if task == 'NLI':
#       sentences = [sentence[0] + " [SEP] " + sentence[1] for sentence in sentences]
#       top_labels=3

#     for i, sentence in enumerate(sentences):

#       # for e-cos dataset, labels change for each sentence. class_names_list parameter should contain the list aof labels for each sentence
#       if class_names_list is not None:
#         self.explainer = LimeTextExplainer(class_names=class_names_list[i], random_state=self.random_state)
#         top_labels= len(class_names_list[i])

#       # if no specific number is given then set num_feature to the number of tokens
#       if num_features is None:
#         num_features_temp = len(sentence.split())
#       else:
#         num_features_temp = num_features

#       explanation = self.explainer.explain_instance(sentence, predict, num_samples=num_samples, num_features=num_features_temp, top_labels=top_labels)
#       explanations.append(explanation)

#     return explanations

#   def show_lime(self, explanations, show_all_labels=False):
    
#     for explanation in explanations:
#       if show_all_labels:
#         label = None
#       else:
#         label = [explanation.top_labels[0]]

#       explanation.show_in_notebook(text=True, labels=label)
#       print('-'*100)
