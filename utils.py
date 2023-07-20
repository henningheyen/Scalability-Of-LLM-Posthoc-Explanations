import numpy as np
import random

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
    #random.seed(seed)  # Set a seed for reproducibility
    #sampled_indices = random.sample(range(len(train_sentences)), k=n_shots)

    # Separate positive and negative samples
    positive_indices = [i for i, label in enumerate(train_labels) if label == 1]
    negative_indices = [i for i, label in enumerate(train_labels) if label == 0]

    # Randomly sample half of the shots from each class
    random.seed(seed)  # Set a seed for reproducibility
    half_n_shots = n_shots // 2
    sampled_indices = random.sample(positive_indices, k=half_n_shots) + random.sample(negative_indices, k=half_n_shots)


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
