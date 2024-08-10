from transformers import AutoModelForSequenceClassification, AutoTokenizer, AdamW
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score

# Sample dataset
data = [
    ("Education is key to sustainable development.", 4),
    ("Climate change impacts every aspect of our lives.", 13),
    ("Affordable and clean energy is vital for progress.", 7),
    ("Zero hunger can be achieved with sustainable agriculture.", 2),
    ("Good health and well-being are fundamental for a prosperous society.", 3),
    ("Gender equality is essential for societal progress.", 5),
    ("Clean water and sanitation ensure healthy communities.", 6),
    ("Decent work and economic growth foster prosperity for all.", 8),
    ("Industry innovation and infrastructure drive economic growth.", 9),
    ("Reducing inequalities promotes social stability.", 10),
    ("Sustainable cities and communities enhance quality of life.", 11),
    ("Responsible consumption and production minimize environmental impact.", 12),
    ("Climate action is necessary to combat global warming.", 13),
    ("Life below water must be protected for future generations.", 14),
    ("Life on land is crucial for biodiversity and ecosystems.", 15),
    ("Peace, justice, and strong institutions promote inclusive societies.", 16),
    ("Partnerships for the goals facilitate sustainable development.", 17)
]


texts, labels = zip(*data)

# Load pre-trained model and tokenizer
model_name = "distilbert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=17)
tokenizer = AutoTokenizer.from_pretrained(model_name)

class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in tokenizer(self.texts[idx], padding='max_length', max_length=512, truncation=True, return_tensors='pt').items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

# Prepare DataLoader
dataset = TextDataset(texts, labels)
loader = DataLoader(dataset, batch_size=2, shuffle=True)

# Tokenize inputs for initial test
inputs = tokenizer(list(texts), padding=True, truncation=True, return_tensors="pt")

# Initial Predictions
outputs = model(**inputs)
initial_predictions = torch.argmax(outputs.logits, dim=-1)
print("Initial Predictions:", initial_predictions.numpy())

# Set up optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Fine-tuning loop
model.train()
for epoch in range(3):  # Number of epochs can be adjusted
    for batch in loader:
        optimizer.zero_grad()
        outputs = model(**{k: v.squeeze() for k, v in batch.items()})
        loss = outputs.loss
        loss.backward()
        optimizer.step()

print("Model fine-tuning completed.")

# Evaluate model
model.eval()
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)

accuracy = accuracy_score(labels, predictions.numpy())
print("Model Accuracy:", accuracy)
