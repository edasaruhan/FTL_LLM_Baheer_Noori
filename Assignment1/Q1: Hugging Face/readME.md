# SDG Text Classification

This project uses Hugging Face's Transformers library to classify text related to the United Nations Sustainable Development Goals (SDGs). It utilizes a pre-trained model, `distilbert-base-uncased`, fine-tuned on a small dataset to predict which SDG a given text relates to.

## Project Objectives

- **Understand** and use Hugging Face's pre-trained models for NLP tasks.
- **Classify** text into categories based on SDGs.
- **Evaluate** model performance before and after fine-tuning.

## Dataset

The dataset consists of text samples related to each of the 17 SDGs. The labels correspond to the SDG number the text is associated with. The dataset is intentionally small for demonstration purposes.

## Installation

To run this project, you'll need to have Python installed, along with the following packages:

```bash
pip install transformers torch scikit-learn
