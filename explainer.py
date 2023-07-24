from lime.lime_text import LimeTextExplainer

class Explainer:
  
  def __init__(self, class_names):
    self.class_names = class_names
    self.explainer = LimeTextExplainer(class_names=class_names)

  def compute_explanations(self, sentences, predict, num_samples=100, num_features=10, task=None):
    explanations = []

    # NLI sentences are pairs of premises and hypotheses
    if task == 'NLI':
      sentences = [sentence[0] + " [SEP] " + sentence[1] for sentence in sentences]

    for sentence in sentences: 
      explanation = self.explainer.explain_instance(sentence, predict, num_samples=num_samples, num_features=num_features)
      explanations.append(explanation)

    return explanations

  def show_lime(self, explanations):

    for explanation in explanations:
      explanation.show_in_notebook(text=True)
      print('-----------------------')

  def comprehensiveness(self, sentence):
  return

  def aggregate_comprehensiveness(self, sentence):
    return

  def sufficiency(self, sentence):
    return

  def aggregate_sufficiency(self, sentence):
    return
