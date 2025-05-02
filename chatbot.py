import json
import pickle
import numpy as np
import random
import spacy
import re

with open('model/chatbot_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('model/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('model/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

with open('data/intents.json') as f:
    intents = json.load(f)

nlp = spacy.load("en_core_web_sm")

def preprocess_input(sentence):
    doc = nlp(sentence)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return ' '.join(tokens)

def get_intent(user_input):
    cleaned_input = preprocess_input(user_input)
    vectorized_input = vectorizer.transform([cleaned_input])
    predicted_class = model.predict(vectorized_input)
    intent = label_encoder.inverse_transform(predicted_class)[0]
    return intent

def get_response(intent_tag):
    for intent in intents['intents']:
        if intent['tag'] == intent_tag:
            return random.choice(intent['responses'])
    return "I'm sorry, I didn't understand that."

def chat():
    print("Bot: Hi! I’m your College Admission Assistant. Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit']:
            print("Bot: Goodbye and good luck with your admissions!")
            break
        intent = get_intent(user_input)
        response = get_response(intent)
        print("Bot:", response)

if __name__ == "__main__":
    chat()
