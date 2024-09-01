import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import gensim
from gensim.models import Word2Vec
import torch
from transformers import BertTokenizer, BertModel
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load your dataset
data = pd.read_csv('your_dataset.csv')
texts = data['text_column']
labels = data['label_column']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Word2Vec Embedding
word2vec_model = Word2Vec(sentences=[text.split() for text in X_train], vector_size=100, window=5, min_count=1, workers=4)

def get_avg_word2vec(text):
    words = text.split()
    embeddings = [word2vec_model.wv[word] for word in words if word in word2vec_model.wv]
    if len(embeddings) > 0:
        return np.mean(embeddings, axis=0)
    else:
        return np.zeros(100)

X_train_word2vec = np.array([get_avg_word2vec(text) for text in X_train])
X_test_word2vec = np.array([get_avg_word2vec(text) for text in X_test])

clf_word2vec = make_pipeline(StandardScaler(), LogisticRegression())
clf_word2vec.fit(X_train_word2vec, y_train)

y_pred_word2vec = clf_word2vec.predict(X_test_word2vec)
print("Word2Vec Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred_word2vec))
print("Precision:", precision_score(y_test, y_pred_word2vec, average='weighted'))
print("Recall:", recall_score(y_test, y_pred_word2vec, average='weighted'))
print("F1-Score:", f1_score(y_test, y_pred_word2vec, average='weighted'))

# GloVe Embedding
glove_embeddings = {}
with open('glove.6B.100d.txt', 'r') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype='float32')
        glove_embeddings[word] = vector

def get_avg_glove(text):
    words = text.split()
    embeddings = [glove_embeddings[word] for word in words if word in glove_embeddings]
    if len(embeddings) > 0:
        return np.mean(embeddings, axis=0)
    else:
        return np.zeros(100)

X_train_glove = np.array([get_avg_glove(text) for text in X_train])
X_test_glove = np.array([get_avg_glove(text) for text in X_test])

clf_glove = make_pipeline(StandardScaler(), LogisticRegression())
clf_glove.fit(X_train_glove, y_train)

y_pred_glove = clf_glove.predict(X_test_glove)
print("GloVe Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred_glove))
print("Precision:", precision_score(y_test, y_pred_glove, average='weighted'))
print("Recall:", recall_score(y_test, y_pred_glove, average='weighted'))
print("F1-Score:", f1_score(y_test, y_pred_glove, average='weighted'))

# BERT Embedding
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

X_train_bert = np.array([get_bert_embedding(text) for text in X_train])
X_test_bert = np.array([get_bert_embedding(text) for text in X_test])

clf_bert = make_pipeline(StandardScaler(), LogisticRegression())
clf_bert.fit(X_train_bert.reshape(len(X_train_bert), -1), y_train)

y_pred_bert = clf_bert.predict(X_test_bert.reshape(len(X_test_bert), -1))
print("BERT Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred_bert))
print("Precision:", precision_score(y_test, y_pred_bert, average='weighted'))
print("Recall:", recall_score(y_test, y_pred_bert, average='weighted'))
print("F1-Score:", f1_score(y_test, y_pred_bert, average='weighted'))

# Optional: Semantic Search Engine
from sklearn.neighbors import NearestNeighbors
import pickle

# Assume 'corpus' is a list of documents
corpus_embeddings = np.array([get_bert_embedding(doc) for doc in corpus])

vector_db = NearestNeighbors(n_neighbors=5, metric='cosine')
vector_db.fit(corpus_embeddings)

with open('vector_db.pkl', 'wb') as f:
    pickle.dump(vector_db, f)

def search(query, vector_db, corpus):
    query_embedding = get_bert_embedding(query)
    distances, indices = vector_db.kneighbors([query_embedding])
    return [corpus[i] for i in indices[0]]

query = "climate change impact"
results = search(query, vector_db, corpus)

print("Search Results for Query:", query)
for i, result in enumerate(results):
    print(f"{i+1}. {result}")
