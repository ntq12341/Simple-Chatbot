import random
import pickle
import nltk
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

# Load model
import os
model_path = os.path.join(os.path.dirname(__file__), "model.pkl")

with open(model_path, "rb") as f:
    model, vectorizer, encoder, data = pickle.load(f)

def clean_input(sentence):
    words = nltk.word_tokenize(sentence.lower())
    words = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(words)

def get_response(user_input):
    cleaned = clean_input(user_input)
    X = vectorizer.transform([cleaned])
    pred = model.predict(X)[0]
    tag = encoder.inverse_transform([pred])[0]

    for intent in data['intents']:
        if intent['tag'] == tag:
            if intent['responses']:
                return random.choice(intent['responses'])
    
    # fallback
    for intent in data['intents']:
        if intent['tag'] == "noanswer":
            return random.choice(intent['responses'])

    return "I don't understand."
