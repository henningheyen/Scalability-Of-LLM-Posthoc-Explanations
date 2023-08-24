import numpy as np
import random
import pandas as pd
from datasets import load_dataset


def make_prompt(dataset, n_shots, test_review):

    prefix = 'This is a movie review dataset with associated positive or negative sentiment: \n'

    sentences = dataset['train'][0:n_shots]['sentence']
    labels = dataset['train'][0:n_shots]['label'] # negative (0) or positive (1)
    labels_str = ['negative' if label == 0 else 'positive' for label in labels]

    shots = ""

    for sentence, label in zip(sentences, labels_str):
        shots = shots + "'" + sentence + "'" + ' - ' + label + ', \n '

    return prefix + shots + "'" + test_review + "'" + ' -'

def make_random_prompt(dataset, n_shots, test_review, seed=42):

    prefix = 'This is a movie review dataset with associated positive or negative sentiment: \n'

    # Using the original train set for the shots (67349 instances)
    train_sentences = dataset['train']['sentence'] 
    train_labels = dataset['train']['label']

    # Randomly sample 20 sentences
    random.seed(seed)  # Set a seed for reproducibility
    sampled_indices = random.sample(range(len(train_sentences)), k=n_shots)

    # Separate positive and negative samples
    positive_indices = [i for i, label in enumerate(train_labels) if label == 1]
    negative_indices = [i for i, label in enumerate(train_labels) if label == 0]

    # Randomly sample half of the shots from each class
    #random.seed(seed)  # Set a seed for reproducibility
    half_n_shots = n_shots // 2

    # making sure that positive and negative reviews alternate
    #negative_indices_sample = random.sample(negative_indices, k=half_n_shots)
    #positive_indices_sample = random.sample(positive_indices, k=half_n_shots)
    
    #sampled_indices = [elem for pair in zip(negative_indices_sample, positive_indices_sample) for elem in pair]
    #sampled_indices = sampled_indices + negative_indices_sample[len(positive_indices_sample):] + positive_indices_sample[len(negative_indices_sample):]

    sampled_indices = random.sample(negative_indices, k=half_n_shots) + random.sample(positive_indices, k=half_n_shots)
    
    # Shuffeling the order of positive and negative shots
    random.shuffle(sampled_indices)
    
    # Retrieve the sampled sentences and labels
    sampled_sentences = [train_sentences[i] for i in sampled_indices]
    sampled_labels = [train_labels[i] for i in sampled_indices]
    sampled_labels_str = ['negative' if label == 0 else 'positive' for label in sampled_labels]

    shots = ""

    for sentence, label in zip(sampled_sentences, sampled_labels_str):
        shots = shots + "'" + sentence + "'" + ' - ' + label + ', \n '


    return prefix + shots + "'" + test_review + "'" + ' -'



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

def predict2(sentences):
  candidate_labels=["negative sentiment", "positive sentiment"]
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

def get_test_set(dataset, test_size, n_shots, seed=42):
    
  if test_size > 872:
      raise ValueError('test_size can be at most 872')

  # Using the original validation set for the test data (872 instances)
  test_sentences = dataset['validation']['sentence']
  test_labels = dataset['validation']['label']

  # Randomly sample test_size number of sentences
  random.seed(seed) 
  sampled_indices = random.sample(range(len(test_sentences)), k=test_size)

  # Retrieve the sampled sentences and labels
  sampled_test_reviews = [test_sentences[i] for i in sampled_indices]
  sampled_labels = [test_labels[i] for i in sampled_indices]
  #sampled_labels_str = ['negative' if label == 0 else 'positive' for label in sampled_labels]

  prompts = [make_random_prompt(dataset, n_shots, test_review, seed) for test_review in sampled_test_reviews]

  return {'prompts': prompts, 'labels': sampled_labels}

def make_test_set(size, dataset_name='MNLI', seed=42):

  # Setting hypothesis and premise identifier
  if dataset_name == 'MNLI':
      dataset = load_dataset('SetFit/mnli')
      premise, hypothesis = 'text1', 'text2'
  elif dataset_name == 'SNLI':
      dataset = load_dataset('snli')
      premise, hypothesis = 'premise', 'hypothesis'
  else:
      raise ValueError('dataset_name must be MNLI or SNLI')

  # Relabeling accoring to deberta model
  new_labels = [2 if label == 1 else 1 if label == 0 else 0 for label in dataset['validation']['label']]

  # Randomly sample the desired number of indices
  random.seed(seed)
  random_indices = random.sample(list(range(len(dataset['validation']))), size)

  test_set = [(dataset['validation'][i][premise], dataset['validation'][i][hypothesis]) for i in random_indices]
  test_labels = [new_labels[i] for i in random_indices]  
  test_labels_text = ['contradiction' if test_labels[i]==0 else 'entailment' if test_labels[i]==1 else 'neutral' for i in range(len(test_labels))]

  return {'sentence_pairs': test_set, 'test_labels': test_labels, 'test_labels_text': test_labels_text}


#deberta: ['contradiction', 'entailment', 'neutral']
#mnli: ['entailment', 'neutral', 'contradiction']
#snli: ['entailment', 'neutral', 'contradiction']

def make_test_set_cose(size, seed=42): # 1221 max size for mnli

  dataset = load_dataset("cos_e", 'v1.11')['validation']

  random.seed(seed)
  random_indices = random.sample(list(range(len(dataset['question']))), size)

  test_set = dataset[random_indices]
  candidate_labels_list = test_set['choices']

  true_labels = [test_set['choices'][i].index(test_set['answer'][i]) for i in range(size)]

  return {
      'question': test_set['question'],
      'choices': test_set['choices'],
      'answer': test_set['answer'],
      'true_labels': true_labels,
      'extractive_explanation': test_set['extractive_explanation'],
      'candidate_labels_list': candidate_labels_list,
  }

def make_test_set_mnli(size, seed=42): # 9815 max size for mnli

  dataset = load_dataset("multi_nli")['validation_matched']
  new_labels = [2 if label == 1 else 1 if label == 0 else 0 for label in dataset['label']] # original: [entailment:0,neutral:1,contradiction:2] deberta: [contradiction:0,entailment:1,neutral:2]

  # Randomly sample the desired number of indices
  random.seed(seed)
  random_indices = random.sample(list(range(len(dataset['label']))), size)

  test_set = [(dataset['premise'][i], dataset['hypothesis'][i]) for i in random_indices]
  print('check3')
  test_labels = [new_labels[i] for i in random_indices]  
  test_labels_text = ['contradiction' if test_labels[i]==0 else 'entailment' if test_labels[i]==1 else 'neutral' for i in range(len(test_labels))]

  return {
      'sentence_pairs': test_set, 
      'test_labels': test_labels, 
      'test_labels_text': test_labels_text
      }

def make_test_set_esnli(size, path='data/esnli_dev.csv', seed=42): # 9842 max size for mnli

  #loading data
  df = pd.read_csv(path)
  df = df[['gold_label', 'Sentence1', 'Sentence2', 'Sentence1_marked_1', 'Sentence2_marked_1', 'Sentence1_Highlighted_1', 'Sentence2_Highlighted_1']]

  dataset = df.to_dict(orient='list')

  random.seed(seed)
  random_indices = random.sample(list(range(df.shape[0])), size)

  sentence_pairs = [(dataset['Sentence1'][i], dataset['Sentence2'][i]) for i in random_indices]
  test_labels_text = [dataset['gold_label'][i] for i in random_indices]
  label_to_num = {'contradiction': 0, 'entailment': 1, 'neutral': 2}   
  test_labels = [label_to_num[label] for label in test_labels_text]
  sentence1_marked = [dataset['Sentence1_marked_1'][i] for i in random_indices]
  sentence2_marked = [dataset['Sentence2_marked_1'][i] for i in random_indices]
  sentence1_highlights = [dataset['Sentence1_Highlighted_1'][i] for i in random_indices]
  sentence2_highlights = [dataset['Sentence2_Highlighted_1'][i] for i in random_indices]


  return {
      'sentence_pairs': sentence_pairs, 
      'test_labels': test_labels, 
      'test_labels_text': test_labels_text,
      'sentence1_marked': sentence1_marked,
      'sentence2_marked': sentence2_marked,
      'sentence1_highlights': sentence1_highlights,
      'sentence2_highlights': sentence2_highlights,
      }