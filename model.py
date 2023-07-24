from transformers import pipeline, GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

class ZeroShotLearner:
    def __init__(self, model_name, candidate_labels):
        self.classifier = pipeline("zero-shot-classification", model=model_name, use_fast=False)
        self.candidate_labels = candidate_labels

    def predict(self, sentences):
        results = self.get_results(sentences)

        # Get the labels from the results
        labels = [pred['labels'] for pred in results]

        # Get the scores from the results
        scores = [pred['scores'] for pred in results]

        # Create NumPy array
        arr = np.zeros((len(sentences), 2))

        # Assign the scores to the corresponding positions in the array
        for i, label in enumerate(labels):
            arr[i, 0] = scores[i][label.index(self.candidate_labels[0])]
            arr[i, 1] = scores[i][label.index(self.candidate_labels[1])]
        
        return arr

    def get_results(self, sentences):
        return self.classifier(sentences, self.candidate_labels)

    def get_predictions(self, results):
        predicted_labels = [result['labels'][0] for result in results]
        mapped_labels = [1 if label == self.candidate_labels[1] else 0 for label in predicted_labels]

        return mapped_labels

class ZeroShotNLI: 
    def __init__(self, model_name):
        self.classifier = AutoModelForSequenceClassification.from_pretrained(f'cross-encoder/{model_name}')
        self.tokenizer = AutoTokenizer.from_pretrained(f'cross-encoder/{model_name}', use_fast= False)

    def predict_for_lime(self, sentence_pairs):

        features = self.tokenizer(sentence_pairs, padding=True, truncation=True, return_tensors="pt")

        self.classifier.eval()
        with torch.no_grad():
            scores = self.classifier(**features).logits # 0: 'contradiction', 1: 'entailment', 0: 'neutral'
            return torch.nn.functional.softmax(scores, dim=-1).detach().numpy()

            


    
class FewShotLearner:
    def __init__(self, model_name):
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    def generate_text(self, prompt, n_generated_tokens=2):
    
        # Tokenize the prompt
        input_ids = self.encode(prompt)

        # Perform a single inference step
        output = self.model.generate(input_ids, max_length=input_ids.size(1) + n_generated_tokens, num_return_sequences=1)

        # Extract and decode the generated token
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)

        # Print the generated token
        print("Generated Text:", generated_text)
        print("Last word: ", generated_text.split()[-1])

    def predict_token(self, prompt, top_k=5, verbose=True):

        # Tokenize the prompt
        input_ids = self.encode(prompt)

        # Perform a single inference step
        output = self.model.forward(input_ids)

        probs = torch.nn.functional.softmax(output.logits[0,-1,:], dim=0)

        # Get the top 5 tokens and their probabilities
        top_probs, top_indices = torch.topk(probs, k=top_k)

        # Convert the indices to tokens
        tokens = self.tokenizer.convert_ids_to_tokens(top_indices.tolist())

        if verbose:
            # Print the top 5 tokens and their probabilities
            for i in range(top_k):
                print(tokens[i], top_probs[i].item())

        return probs
    
    def predict_for_lime(self, prompts):

        results = []

        for prompt in prompts:

            probs = self.predict_token(prompt, verbose=False)

            token_probs = probs[self.tokenizer.convert_tokens_to_ids(['Ġnegative', 'Ġpositive'])] # 3967: id for Ġpositive, 4633: id for Ġnegative
            
            norm_probs = token_probs / torch.sum(token_probs) # normalizing

            results.append([norm_probs[0].item(), norm_probs[1].item()])

        return np.array(results)
    
    def get_predictions(self, prompts):
        # Outputs 0 for negative and 1 for positive prediction

        predicted_labels = []
        
        for prompt in prompts:
            probs = self.predict_token(prompt, verbose=False)

            if torch.argmax(probs) == 4633: # Ġnegative
                predicted_labels.append(0)
            elif torch.argmax(probs) == 3967: # Ġpositive
                predicted_labels.append(1)
            else:
                # Neither Ġpositive nor Ġnegative was predicted
                predicted_token = self.tokenizer.convert_ids_to_tokens(int(torch.argmax(probs)))
                raise ValueError(f'Predicted token is {predicted_token} instead of Ġpositive or Ġnegative')
            
        return predicted_labels


    
    def encode(self, prompt):
        return self.tokenizer.encode(prompt, return_tensors='pt')

    
