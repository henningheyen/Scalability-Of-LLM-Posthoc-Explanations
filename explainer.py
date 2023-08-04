from lime.lime_text import LimeTextExplainer
import numpy as np
import re
import string



class Explainer:
  
  def __init__(self, class_names=None, random_state=42):
    self.class_names = class_names
    self.random_state = random_state
    self.explainer = LimeTextExplainer(class_names=class_names, random_state=random_state)

  def compute_explanations(self, sentences, model, num_samples=100, num_features=None, task=None, class_names_list=None):
    explanations = []

    # NLI sentences are pairs of premises and hypotheses
    if task == 'NLI':
      sentences = [sentence[0] + " [SEP] " + sentence[1] for sentence in sentences]
      top_labels=3

    # making pertubations deterministic
    np.random.seed(random_state)

    for i, sentence in enumerate(sentences):

      # for e-cos dataset, labels change for each sentence. class_names_list parameter should contain the list aof labels for each sentence
      if class_names_list is not None:
        self.explainer = LimeTextExplainer(class_names=class_names_list[i], random_state=self.random_state)
        top_labels= len(class_names_list[i])
      else:
         class_names_list = [self.class_names]
         top_labels = len(self.class_names)

      # if no specific number is given then set num_feature to the number of tokens
      if num_features is None:
        num_features_temp = len(sentence.split())
      else:
        num_features_temp = num_features

      explanation = self.explainer.explain_instance(
         sentence, 
         lambda x: model.predict(x, [class_names_list[i]]*num_features_temp), 
         num_samples=num_samples, 
         num_features=num_features_temp, 
         top_labels=top_labels
         )
      explanations.append(explanation)

    return explanations

  def show_lime(self, explanations, show_all_labels=False):
    
    for explanation in explanations:
      if show_all_labels:
        label = None
      else:
        label = [explanation.top_labels[0]]
      
      explanation.show_in_notebook(text=True, labels=label)
      print('-'*100)


  def comprehensiveness(self, explanation, sentence_pair, predict, top_k=None, top_percent=None, verbose=True):
      
      # Rationale is the top_k tokens 
      rationale = self.get_ratinoale(explanation, top_k=top_k, top_percent=top_percent) 

      # Removing the rationale from the original sentence
      tokens = (sentence_pair[0] + " [SEP] " + sentence_pair[1]).split()
      tokens_minus_rationale = self.remove_top_lime_tokens(rationale, tokens)
      sentence_new = ' '.join(tokens_minus_rationale)

      # Computing new probability for predicted class
      predicted_class_index = np.argmax(explanation.predict_proba) # predicted class 0: 'contradiction', 1: 'entailment', 2: 'neutral'
      prediction_new = predict([sentence_new])[0]
      probability_new = prediction_new[predicted_class_index]

      # Comparing predictions (with and without rationale)
      probability_old = max(explanation.predict_proba) #for the predicted class
      comprehensiveness = probability_old - probability_new

      # Print statement
      if verbose:
          print('rationale: ', rationale)
          print('sentence_old: ', (sentence_pair[0] + " [SEP] " + sentence_pair[1]))    
          print('sentence_new: ', sentence_new)    
          print('probability_old: ', probability_old)
          print('predicted label_old: ', self.explainer.class_names[explanation.top_labels[0]])
          print('probability_new: ', probability_new)    
          print('predicted label_new: ', self.explainer.class_names[np.argmax(prediction_new)])
          print('comprehensiveness: ', comprehensiveness, '\n')    

      return comprehensiveness

  def aggregated_metric(self, metric, explanation, sentence_pair, predict, bins=[0.1, 0.3, 0.5], verbose=False):

    if metric not in ['comprehensiveness', 'sufficiency']:
      raise TypeError("The 'metric' parameter must either 'comprehensiveness' or 'sufficiency'")

    aggregate = []

    for top_percent in bins:
      if metric == 'comprehensiveness':
        comp = self.comprehensiveness(explanation, sentence_pair, predict, top_percent=top_percent, verbose=verbose)
        aggregate.append(comp)    
      else:
        suff = self.sufficiency(explanation, sentence_pair, predict, top_percent=top_percent, verbose=verbose)
        aggregate.append(suff)    
    return np.mean(aggregate)

  def sufficiency(self, explanation, sentence_pair, predict, top_k=None, top_percent=None, verbose=True):

    # Rationale is the top_k tokens 
    rationale = self.get_ratinoale(explanation, top_k=top_k, top_percent=top_percent) 

    # Forming new_sentence from rationale while retaining order from old_sentence
    tokens = (sentence_pair[0] + " [SEP] " + sentence_pair[1]).split()
    sentence_new = ' '.join(token for token in tokens if token in rationale or token == "[SEP]")

    predicted_class_index = np.argmax(explanation.predict_proba) # predicted class 0: 'contradiction', 1: 'entailment', 2: 'neutral'
    prediction_new = predict([sentence_new])[0]
    probability_new = prediction_new[predicted_class_index]

    # Comparing predictions (with and without rationale)
    probability_old = max(explanation.predict_proba) #for the predicted class

    sufficiency = probability_old - probability_new

    if verbose:
        print('rationale: ', rationale)
        print('sentence_old: ', (sentence_pair[0] + " [SEP] " + sentence_pair[1]))    
        print('sentence_new: ', sentence_new)    
        print('probability_old: ', probability_old)
        print('predicted label_old: ', self.explainer.class_names[explanation.top_labels[0]])
        print('probability_new: ', probability_new)    
        print('predicted label_new: ', self.explainer.class_names[np.argmax(prediction_new)])
        print('sufficiency: ', sufficiency, '\n')

    return sufficiency

  def get_ratinoale(self, explanation, top_k=None, top_percent=None):
      expl_list = explanation.as_list(label= explanation.top_labels[0])

      if top_k is not None:
          threshold = top_k
      elif top_percent is not None:
          threshold = int(np.ceil(len(expl_list) * top_percent))
      else:
          # if neither top_k nor top_percent is set then return comprehensiveness for top_k=3 rationale
          threshold = 3

      rationale = [token_score_pair[0] for token_score_pair in expl_list[:threshold]]
      return rationale
      

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

  # def __init__(self, class_names=None, random_state=42):
  #   self.class_names = class_names
  #   self.random_state = random_state
  #   self.explainer = LimeTextExplainer(class_names=class_names, random_state=random_state)

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

  # def show_lime(self, explanations, show_all_labels=False):
    
  #   for explanation in explanations:
  #     if show_all_labels:
  #       label = None
  #     else:
  #       label = [explanation.top_labels[0]]

  #     explanation.show_in_notebook(text=True, labels=label)
  #     print('-'*100)