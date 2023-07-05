from lime.lime_text import LimeTextExplainer


def show_lime(sentences, predict):

  class_names = ['negative sentiment', 'positive sentiment']
  explainer = LimeTextExplainer(class_names=class_names)
  
  for sentence in sentences:
    explanation = explainer.explain_instance(sentence, predict, num_samples=100)
    print("Sentence: ", sentence)
    print("LIME Explanation:")
    explanation.show_in_notebook(text=True)
    print('-----------------------')

