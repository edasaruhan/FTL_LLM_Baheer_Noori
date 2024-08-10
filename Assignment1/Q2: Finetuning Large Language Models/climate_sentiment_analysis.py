import pandas as pd
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Sample dataset creation (replace with actual dataset loading)
data = {
    "text": [
        "Climate change is the most pressing issue of our time.",
        "Renewable energy sources are crucial for reducing emissions.",
        "The government's climate policy is inadequate.",
        "Planting trees is an effective way to combat climate change.",
        "There is no evidence that climate change is caused by humans.",
        "Climate action requires immediate attention from all nations."
    ],
    "label": [1, 1, 0, 1, 0, 1]  # 1: Positive, 0: Negative
}

# Convert to DataFrame
df = pd.DataFrame(data)

# EDA: Visualize data distribution
sns.countplot(x='label', data=df)
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()

# Train-test split
train_texts, test_texts, train_labels, test_labels = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# Load pre-trained model and tokenizer
model_name = "bert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenization
train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(list(test_texts), truncation=True, padding=True, max_length=128)

class ClimateDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = ClimateDataset(train_encodings, train_labels)
test_dataset = ClimateDataset(test_encodings, test_labels)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    warmup_steps=10,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

# Define a compute_metrics function for evaluation
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    return {
        'accuracy': acc,
        'f1': f1,
    }

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# Evaluate before fine-tuning
print("Initial Evaluation:", trainer.evaluate())

# Fine-tune the model
trainer.train()

# Evaluate after fine-tuning
print("Post-Fine-Tuning Evaluation:", trainer.evaluate())
