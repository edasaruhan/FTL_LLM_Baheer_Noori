
### report.md

```markdown
# SDG Text Classification Project Report

## Introduction

This project aims to classify text into categories representing the United Nations Sustainable Development Goals (SDGs)
using the Hugging Face Transformers library. By leveraging a pre-trained NLP model, we can efficiently categorize text
related to various SDGs, facilitating better organization and prioritization of sustainable development content.

## Objectives

1. Familiarize with the Hugging Face library and its capabilities.
2. Implement a text classification model that identifies text related to SDGs.
3. Evaluate the model's performance before and after fine-tuning.

## Methodology

### Data Preparation

A synthetic dataset was created, consisting of short text samples labeled according to the SDG they relate to.
Each sample text was paired with an integer label representing one of the 17 SDGs.

### Model Selection

The `distilbert-base-uncased` model was selected due to its balance between performance and efficiency.
This model is well-suited for text classification tasks and provides a good starting point for fine-tuning on specific datasets.

### Implementation Steps

1. **Loading the Model**: The pre-trained model and tokenizer were loaded from the Hugging Face model hub.
2. **Data Tokenization**: The text data was tokenized and prepared for input into the model.
3. **Initial Evaluation**: The model's predictions were evaluated before any fine-tuning to establish a performance baseline.
4. **Fine-Tuning**: The model was fine-tuned on the dataset using a simple training loop with the AdamW optimizer.
5. **Final Evaluation**: The model's performance was evaluated again after fine-tuning to assess improvements.

### Evaluation Metrics

Accuracy was used as the primary metric to evaluate the model's performance, comparing predictions against true labels.

## Results

The fine-tuned model showed an improvement in accuracy over the initial, pre-fine-tuning model, demonstrating the effectiveness
of fine-tuning for specific tasks like SDG text classification.

## Conclusion

This project successfully demonstrates how to use Hugging Face's Transformers library to build and fine-tune a text classification
model for SDG-related text. Further improvements can be achieved by expanding the dataset and exploring other model architectures
or fine-tuning techniques.

## Future Work

- **Expand the Dataset**: Acquire a larger and more diverse dataset for better model generalization.
- **Explore Advanced Techniques**: Investigate other models and fine-tuning strategies to improve performance.
- **Real-World Application**: Apply the model to real-world data sources to automatically classify and tag content related to SDGs.

## References

- Hugging Face Transformers: [https://huggingface.co/transformers/](https://huggingface.co/transformers/)
- United Nations Sustainable Development Goals: [https://sdgs.un.org/goals](https://sdgs.un.org/goals)
