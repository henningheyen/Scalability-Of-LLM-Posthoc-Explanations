# Transformers Explainability

This repository contains the code and experiments related to my Master Thesis "Explainability in Transformer-Based Pre-Trained Language Model". The code conducts experiments on how explainability is affected by model size. 

We use LIME explanations and DeBERTaV3 models of four different sizes from Huggingface. The project investigates natural language inference (NLI) and zero shot classification (ZSC). We use three datasets (MNLI, e-SNLI and CoS-e). The explanations are evaluated based on their faithfulness and plausibility. While faithfulness aims to capture the extend to which the explanation reflects the model's internal decision process, plausibility captures the agreement between the generated explanation and a human ground truth explanation. Faithfulness is measured by comprehensiveness and sufficiency and plausibility is measured by intersection over union (IOU) and token level f1 scores (TokenF1). 

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

```bash
pip install -r requirements.txt
```

## Extending the Repository

The repository is designed with extensibility in mind. Here's how you can extend it:

1. **Add Models**: Add other NLI or ZSC models from Huggingface by changing the `model_name` parameter. For other tasks than NLI or ZSC create a new class similar to `ModelZSC` or `ModelNLI` based on your requirements. 

2. **Add Datasets**: Add data loading and preprocessing functions to `utils.py` and integrate them with the existing evaluation pipelines. 

3. **Add Explainability Techniques**: The current state uses LIME in `explainer.py` script. can be extended to incorporate other explainability techniques. Create new classes or methods similar to the existing `Explainer` class to extend by other explainability techniques (e.g. SHAP, Anchors, Integrated Gradients).

4. **Add explainability metrics**: The repository currently uses comprehensiveness and sufficiency (faithfulness) and IOU and TokenF1 (plausibility) as evaluation metrics. You can add more metrics by extending the evaluation notebooks or the `explainer.py` script.


A paper based on this research is being prepared for submission at the NeurIPS 2023 workshop for Socially Responsible Language Modelling Research (SoLaR). We add a reference to this repository once the paper has been reviewed. 
