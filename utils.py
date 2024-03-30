import random
import pandas as pd
from datasets import load_dataset


def make_test_set_cose(size, seed=42, remove_bad_explanations=False): # 1221 max size for mnli, 503 bad explanations
  """
  Imports and loads CoS-e dataset. Uses the validation set. If size is smaller than max size, samples are selected randomly. 
  """

  dataset = load_dataset("cos_e", 'v1.11')['validation']

  if remove_bad_explanations:
    idx = [i for i in range(len(dataset['question'])) if len(dataset['extractive_explanation'][i].split()) < len(dataset['question'][i].split())]
    random.seed(seed)
    random_indices = random.sample(idx, size) 
  else:
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
      'extractive_explanation': [test_set['extractive_explanation'][i].split() for i in range(size)],
      'candidate_labels_list': candidate_labels_list,
      'random_indices': random_indices,
  }

def make_test_set_mnli(size, seed=42): # 9815 max size for mnli
  """
  Imports and loads MNLI dataset. Uses the validation set. If size is smaller than max size, samples are selected randomly. 
  """

  dataset = load_dataset("multi_nli")['validation_matched']
  new_labels = [2 if label == 1 else 1 if label == 0 else 0 for label in dataset['label']] # original: [entailment:0,neutral:1,contradiction:2] deberta: [contradiction:0,entailment:1,neutral:2]

  # Randomly sample the desired number of indices
  random.seed(seed)
  random_indices = random.sample(list(range(len(dataset['label']))), size)
  test_set = [(dataset['premise'][i], dataset['hypothesis'][i]) for i in random_indices]
  test_labels = [new_labels[i] for i in random_indices]  
  test_labels_text = ['contradiction' if test_labels[i]==0 else 'entailment' if test_labels[i]==1 else 'neutral' for i in range(len(test_labels))]

  return {
      'sentence_pairs': test_set, 
      'test_labels': test_labels, 
      'test_labels_text': test_labels_text,
      'random_indices': random_indices,
      }

def make_test_set_esnli(size, path='data/esnli_dev.csv', seed=42, remove_bad_explanations=False): # 9842 max size for e-SNLI
  """
  Imports and loads e-SNLI dataset. Uses the validation set. If size is smaller than max size, samples are selected randomly. 
  """

  #loading data
  df = pd.read_csv(path)
  df = df[['gold_label', 'Sentence1', 'Sentence2', 'Sentence1_marked_1', 'Sentence2_marked_1', 'Sentence1_Highlighted_1', 'Sentence2_Highlighted_1']]

  if remove_bad_explanations: 
    # Remove wrongyl annotated instances
    df.drop(index=[262, 1572, 1642, 2041, 5317, 5779, 5780, 5918, 6973, 7940, 9233, 9305], inplace=True)

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

  # Getting extractive explanatinos. We use the set of highlights accross both premise and hyporthesis:
  extractive_explanations = []
  for i in range(size):

    highlights_1_str = sentence1_highlights[i].split(',')
    highlights_2_str = sentence2_highlights[i].split(',')

    if highlights_1_str[0] == '{}':
      highlights_1_idx = []
    else:
      highlights_1_idx = [int(highlights_1_str[j]) for j in range(len(highlights_1_str))]

    if highlights_2_str[0] == '{}':
      highlights_2_idx = []
    else:
      highlights_2_idx = [int(highlights_2_str[j]) for j in range(len(highlights_2_str))]

    highlights_1_tokens = [sentence_pairs[i][0].split()[j] for j in highlights_1_idx]
    highlights_2_tokens = [sentence_pairs[i][1].split()[j] for j in highlights_2_idx]

    extractive_explanations.append(list(set(highlights_1_tokens + highlights_2_tokens)))

  return {
      'sentence_pairs': sentence_pairs, 
      'test_labels': test_labels, 
      'test_labels_text': test_labels_text,
      'sentence1_marked': sentence1_marked,
      'sentence2_marked': sentence2_marked,
      'sentence1_highlights': sentence1_highlights,
      'sentence2_highlights': sentence2_highlights,
      'extractive_explanation': extractive_explanations,
      'random_indices': random_indices,
      }
      