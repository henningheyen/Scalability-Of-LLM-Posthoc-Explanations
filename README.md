# ICLR Submission

This repository contains the code and experiments related to the paper "The Effect of Model Size on LLM Post-hoc Explainability via LIME". The paper was published in the ICLR 2024 workshop on Secure and Trustworthy Large Language Models ([SeT-LLM](https://set-llm.github.io)). 

## Abstract
Large language models (LLMs) are becoming bigger to boost performance. However, little is known about how explainability is affected by this trend. This work explores LIME explanations for DeBERTaV3 models of four different sizes on natural language inference (NLI) and zero-shot classification (ZSC) tasks. We evaluate the explanations based on their **faithfulness** to the models'  internal decision processes and their **plausibility**, i.e. their agreement with human explanations. Our results suggest some extent of misalignment between the LIME explanations and the models' internal processes as model size increases.  

## Table of Contents

- [Repository Structure](#repository-structure)
- [Setup](#setup)
- [Extending the Repository](#extending-the-repository)

## Repository Structure

- **Notebooks**:
  - `Colab_Model_Evaluation.ipynb`: Colab notebook to evaluate model performance of the validation sets.
  - `Colab_Explainability_Evaluation.ipynb`: Colab notebook for calculating 100 LIME explanations for each dataset. 
  - `explainability_local_evaluation.ipynb`: Applying explainability metrics on local instances.
  - `explainability_global_evaluation_results.ipynb`: Applying the metrics across 100 explanations. 
  
  
- **Python Scripts**:
  - `explainer.py`: Contains the `Explainer` class which is responsible for generating LIME explanations for the models.
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

