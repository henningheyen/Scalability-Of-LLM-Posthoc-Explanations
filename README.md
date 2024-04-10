# Scalability Of LLM Posthoc-Explanations

This repository contains the code and experiments related to the paper "The Effect of Model Size on LLM Post-hoc Explainability via LIME". The paper was published in the ICLR 2024 workshop on Secure and Trustworthy Large Language Models ([SeT-LLM](https://set-llm.github.io)). 

## Abstract
Large language models (LLMs) are becoming bigger to boost performance. However, little is known about how explainability is affected by this trend. This work explores LIME explanations for DeBERTaV3 models of four different sizes on natural language inference (NLI) and zero-shot classification (ZSC) tasks. We evaluate the explanations based on their **faithfulness** to the models’ internal decision processes and their **plausibility**, i.e. their agreement with human explanations. The key finding is that increased model size does not correlate with plausibility despite improved model performance, suggesting a misalignment between the LIME explanations and the models’ internal processes as model size increases. Our results further suggest limitations regarding the comprehensiveness metric for faithfulness in NLI contexts.
  

## Table of Contents

- [Repository Structure](#repository-structure)
- [Setup](#setup)
- [Extending the Repository](#extending-the-repository)

## Repository Structure

- **Notebooks**:
  - `Colab_Model_Evaluation.ipynb`: Colab notebook to evaluate model performance.
  - `Colab_Explainability_Evaluation.ipynb`: Colab notebook for calculating 100 LIME explanations for each dataset and evaluating the explanations based on comprehensiveness and IOU scores
  - `explainability_local_evaluation.ipynb`: Applying explainability metrics on local instances to illustrate the explainability metrics for faithfulness (comprehensiveness) and plausibility (Intersection Over Union or IOU)
  - `plots_tables_results.ipynb`: Visualizing results on performance, comprehensiveness and IOU
  - `model_evaluation_tables.ipynb`: model performance summarized in tables
  
- **Python Scripts**:
  - `explainer.py`: Contains the `Explainer` class which is responsible for generating LIME explanations for the models and evaluating their comprehensiveness and IOU. 
  - `model.py`: Contains the `ModelZSC` and `ModelNLI` classes for zero-shot classification and natural language inference models respectively.
  - `utils.py`: Contains data loader functions for MNLI, e-SNLI and CoS-e
 
- **Results**:
  - `json`: folder that contains JSON formatted results regarding model performance and LIME explanations (to avoid running intense computations again)
  - `figures`: folder that contains main plots associated with the results of this project
  - `tables`: folder with CSV formatted tables containing the results in numbers.    
    
- **Requirements**:
  - `requirements.txt`: Lists the dependencies required to run the code in this repository.
 
- **License**:
  - `LICENSE.txt`: This project is licensed under the MIT License

## Setup

To set up the environment to run the experiments, install the required packages using:

`pip install -r requirements.txt`

## Extending the Repository

The repository is designed with extensibility in mind. Here's how you can extend it:

1. **Add Models**: Add other NLI or ZSC models from Huggingface by changing the `model_name` parameter. For other tasks than NLI or ZSC create a new class similar to `ModelZSC` or `ModelNLI` based on your requirements. 

2. **Add Datasets**: Add data loading and preprocessing functions to `utils.py` and integrate them with the existing evaluation pipelines. 

3. **Add Explainability Techniques**: The current state uses LIME in `explainer.py` script. can be extended to incorporate other explainability techniques. Create new classes or methods similar to the existing `Explainer` class to extend by other explainability techniques (e.g. SHAP, Anchors, Integrated Gradients).

4. **Add explainability metrics**: The repository currently uses comprehensiveness (faithfulness) and IOU (plausibility) as evaluation metrics. You can add more metrics by extending the evaluation notebooks or the `explainer.py` script.

