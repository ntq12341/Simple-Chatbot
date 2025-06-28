import json
import random
import pickle

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')

# Load data
import os
base_dir = os.path.dirname(__file__)
data_path = os.path.join(base_dir, "data", "intents.json")

with open(data_path, "r") as f:
    data = json.load(f)

lemmatizer = WordNetLemmatizer()
texts = []
labels = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        words = nltk.word_tokenize(pattern.lower())
        words = [lemmatizer.lemmatize(w) for w in words]
        texts.append(" ".join(words))
        labels.append(intent['tag'])

# Encode labels
encoder = LabelEncoder()
y = encoder.fit_transform(labels)

# Vectorize
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# Train model
model = LogisticRegression()
model.fit(X, y)


# Save
model_path = os.path.join(os.path.dirname(__file__), "model.pkl")

with open(model_path, "wb") as f:
    pickle.dump((model, vectorizer, encoder, data), f)

print("Model trained and saved.")
