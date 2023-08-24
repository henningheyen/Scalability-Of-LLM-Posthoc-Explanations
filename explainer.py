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
        self.explainer = LimeTextExplainer(class_names=class_names_list[i], random_state=self.random_state)
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


  def comprehensiveness(self, explanation, sentence, predict, top_k=None, top_percent=None, verbose=True, task='NLI'):
      
      if task not in ['NLI', 'ZSC']:
        raise TypeError("Use 'NLI or 'ZSC' for task parameter or extend this method")

      # Rationale is the top_k tokens 
      rationale = self.get_ratinoale(explanation, top_k=top_k, top_percent=top_percent) 

      # Forming new_sentence from rationale while retaining order from old_sentence
      if task == 'NLI': # Natural Language Inference
        tokens = (sentence[0] + " [SEP] " + sentence[1]).split()
      elif task == 'ZSC': # Zero Shot Classification
        tokens = sentence.split()

      # Removing the rationale from the original sentence
      tokens_minus_rationale = [token for token in tokens if token not in rationale or token == '[SEP]'] # [SEP] should never be removed for NLI task
      sentence_new = ' '.join(tokens_minus_rationale)

      # Computing new probability for predicted class
      predicted_class_index = np.argmax(explanation.predict_proba) # predicted class 0: 'contradiction', 1: 'entailment', 2: 'neutral' (NLI)
      prediction_new = predict([sentence_new])[0]
      probability_new = prediction_new[predicted_class_index]

      # Comparing predictions (with and without rationale)
      probability_old = max(explanation.predict_proba) #for the predicted class
      comprehensiveness = probability_old - probability_new

      # Print statement
      if verbose:
          print('rationale: ', rationale)
          print('sentence_old: ', sentence)
          print('sentence_new: ', sentence_new)    
          print('probability_old: ', probability_old)
          print('predicted label_old: ', self.explainer.class_names[explanation.top_labels[0]])
          print('probability_new: ', probability_new)    
          print('predicted label_new: ', self.explainer.class_names[np.argmax(prediction_new)])
          print('comprehensiveness: ', comprehensiveness, '\n')    

      return comprehensiveness


  def sufficiency(self, explanation, sentence, predict, top_k=None, top_percent=None, verbose=True, task='NLI'):

    if task not in ['NLI', 'ZSC']:
      raise TypeError("Use 'NLI or 'ZSC' for task parameter or extend this method")

    # Rationale is the top_k tokens 
    rationale = self.get_ratinoale(explanation, top_k=top_k, top_percent=top_percent) 

    # Forming new_sentence from rationale while retaining order from old_sentence
    if task == 'NLI': # Natural Language Inference
      tokens = (sentence[0] + " [SEP] " + sentence[1]).split()
    elif task == 'ZSC': # Zero Shot Classification
      tokens = sentence.split()
    
    sentence_new = ' '.join(token for token in tokens if token in rationale or token == "[SEP]")

    predicted_class_index = np.argmax(explanation.predict_proba) # predicted class 0: 'contradiction', 1: 'entailment', 2: 'neutral'
    prediction_new = predict([sentence_new])[0]
    probability_new = prediction_new[predicted_class_index]

    # Comparing predictions (with and without rationale)
    probability_old = max(explanation.predict_proba) #for the predicted class

    sufficiency = probability_old - probability_new

    if verbose:
        print('rationale: ', rationale)
        print('sentence_old: ', sentence)    
        print('sentence_new: ', sentence_new)    
        print('probability_old: ', probability_old)
        print('predicted label_old: ', self.explainer.class_names[explanation.top_labels[0]])
        print('probability_new: ', probability_new)    
        print('predicted label_new: ', self.explainer.class_names[np.argmax(prediction_new)])
        print('sufficiency: ', sufficiency, '\n')

    return sufficiency

  def aggregated_metric(self, metric, explanation, sentence, predict, bins=[0.1, 0.3, 0.5], verbose=False, task='NLI'):

    if metric not in ['comprehensiveness', 'sufficiency']:
      raise TypeError("The 'metric' parameter must either 'comprehensiveness' or 'sufficiency'")

    if task not in ['NLI', 'ZSC']:
      raise TypeError("Use 'NLI or 'ZSC' for task parameter or extend this method")

    aggregate = []

    for top_percent in bins:
      if metric == 'comprehensiveness':
        comp = self.comprehensiveness(explanation, sentence, predict, top_percent=top_percent, verbose=verbose, task=task)
        aggregate.append(comp)    
      else:
        suff = self.sufficiency(explanation, sentence, predict, top_percent=top_percent, verbose=verbose, task=task)
        aggregate.append(suff)    
    return np.mean(aggregate)

  def get_ratinoale(self, explanation, top_k=None, top_percent=None):
    expl_list = explanation.as_list(label= explanation.top_labels[0])
    expl_list_sorted = sorted(expl_list, key=lambda x: x[1], reverse=True) # sorting in descending order

    if top_k is not None:
        threshold = top_k
    elif top_percent is not None:
        threshold = int(np.ceil(len(expl_list_sorted) * top_percent))
    else:
        # if neither top_k nor top_percent is set then return comprehensiveness for top_k=3 rationale
        threshold = 3

    rationale = [token_score_pair[0] for token_score_pair in expl_list_sorted[:threshold]]
    return rationale
      
  # NOT used. Review..
  def remove_top_lime_tokens(self, rationale, tokens):
      # ensure [SEP] is never removed
      if '[SEP]' in rationale:
          rationale.remove("[SEP]")
      
      # Create a list for new sentence tokens
      new_sentence = []
      
      for token in tokens:
          # Split the token into words and punctuation
          subtokens = re.findall(r'\w+|[^\w\s]', token)
          
          # Check each subtoken against the rationale
          new_subtokens = [subtoken for subtoken in subtokens if subtoken not in rationale or subtoken == 'SEP' or not re.match(r'\w+', subtoken)]
          
          # Reassemble the token
          new_token = ''.join(new_subtokens)
          if new_token:
              new_sentence.append(new_token)
      
      return new_sentence


  # def lime_tokenize(self, sentence, split_expression=r'\W+'):
  #   """
  #   Tokenizes the sentence based on the provided split_expression same as LIME documentation (splits on non-word characters).
  #   """
  #   # Using the split_expression as a non-capturing group
  #   splitter = re.compile(r'(%s)|$' % split_expression)
  #   return [s for s in splitter.split(sentence) if s]
  
  def get_explanation_list(self, explanations, top_percent):
    
    explanation_list = []
    
    for explanation in explanations:
        expl_list = [explanation.as_list(label= explanation.top_labels[0])[i][0].lower() for i in range(len(explanation.as_list()))]
        threshold = int(np.ceil(len(expl_list) * top_percent))
        explanation_list.append(expl_list[:threshold])
    
    return explanation_list
  
  def format_explanation_true_list(self, explanation_true_list):

    # spliting words with apostrophe ('wasn't' -> 'wasn', 't')
    explanation_true_list = [[word for item in explanation for word in item.replace("'", " ").split()] for explanation in explanation_true_list]

    # Now split the words by spaces and remove punctuation ('kissing?' -> 'kissing')
    translator = str.maketrans('', '', string.punctuation)
        
    # Use a list comprehension to remove punctuation from each word in each list.
    explanation_true_list = [[word.translate(translator) for word in word_list] for word_list in explanation_true_list]

    return explanation_true_list
  
  def compute_token_f1(self, explanation_true, explanation_pred):
    true_set = set(explanation_true)
    pred_set = set(explanation_pred)

    # Calculate precision and recall
    precision = len(true_set & pred_set) / len(pred_set) if pred_set else 0
    recall = len(true_set & pred_set) / len(true_set) if true_set else 0

    # Calculate F1 score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return f1


  def compute_macro_iou(self, rationales, ground_truths, threshold=0.5):
      """
      Computes IoU values for each pair of explanations from two lists and checks if they exceed a threshold.
      
      Args:
      - rationales: List of explanations (lists of tokens).
      - ground_truths: List of explanations (lists of tokens).
      - threshold: IoU threshold to consider a match.
      
      Returns:
      - average of List of fractions indiction the IOU for each instance
      """
      # Check if lists are of the same length
      if len(rationales) != len(ground_truths):
          raise ValueError("Both lists should be of the same length.")
      
      iou_scores = [self.compute_instance_iou(rationale, ground_truth) for rationale, ground_truth in zip(rationales, ground_truths)]
      return np.mean(iou_scores)


  def compute_instance_iou(self ,rationale, ground_truth):
    """Helper function to compute IoU for two explanations."""
    set_rationale = set(rationale)
    set_ground_truth = set(ground_truth)
    intersection = len(set_rationale.intersection(set_ground_truth))
    union = len(set_rationale.union(set_ground_truth))
    return intersection / union if union != 0 else 0


  # similar to ERASER repo: https://github.com/jayded/eraserbenchmark/blob/master/rationale_benchmark/metrics.py 
  def compute_instance_f1(rationale, ground_truth):
      """
      Computes the instance F1 score for two given lists of tokens.
      """
      set_rationale = set(rationale)
      set_ground_truth = set(ground_truth)
      
      tp = len(set_rationale.intersection(set_ground_truth))
      fp = len(set_rationale.difference(set_ground_truth))
      fn = len(set_ground_truth.difference(set_rationale))
      
      precision = tp / (tp + fp) if (tp + fp) != 0 else 0
      recall = tp / (tp + fn) if (tp + fn) != 0 else 0
      f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0
      
      return f1

  def macro_f1_for_lists(rationales, ground_truths):
      """
      Computes the macro F1 score for two given lists of lists of tokens.
      """
      assert len(rationales) == len(ground_truths), "Both lists must have the same length."
      
      f1_scores = [compute_instance_f1(r, gt) for r, gt in zip(rationales, ground_truths)]
      
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