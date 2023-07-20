from lime.lime_text import LimeTextExplainer

class_names = ['negative', 'positive']
explainer = LimeTextExplainer(class_names=class_names)

def show_lime(sentences, predict, num_samples=100, num_features=10):
  
  for sentence in sentences:
    explanation = compute_explanation(sentence, predict, num_samples=num_samples, num_features=num_features)
    print("Sentence: ", sentence)
    print("LIME Explanation:")
    explanation.show_in_notebook(text=True)
    print('-----------------------')

def compute_explanation(sentence, predict, num_samples=100, num_features=10):
  return explainer.explain_instance(sentence, predict, num_samples=num_samples, num_features=num_features)

def comprehensiveness(sentence):
  
  return

def aggregate_comprehensiveness(sentence):
  return

def sufficiency(sentence):
  return

def aggregate_sufficiency(sentence):
  return
