from lime.lime_text import LimeTextExplainer

class_names = ['negative sentiment', 'positive sentiment']
explainer = LimeTextExplainer(class_names=class_names)

def show_lime(sentences, predict):
  
  for sentence in sentences:
    explanation = compute_explanation(sentence, predict, num_samples=100)
    print("Sentence: ", sentence)
    print("LIME Explanation:")
    explanation.show_in_notebook(text=True)
    print('-----------------------')

def compute_explanation(sentence, predict, num_samples=100):
  return explainer.explain_instance(sentence, predict, num_samples=100)

def comprehensiveness(sentence):
  
  return

def aggregate_comprehensiveness(sentence):
  return

def sufficiency(sentence):
  return

def aggregate_sufficiency(sentence):
  return
