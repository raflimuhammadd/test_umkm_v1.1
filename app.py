from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import json
import os
import pandas as pd
import tensorflow as tf
import string
from tensorflow.keras.models import Model
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model
import random

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

with open('./content/umkm.json', encoding='utf-8') as content:
    dataset = json.load(content)

app = Flask(__name__)

tags = []
inputs = []
responses = {}
words = []
classes = []
documents = []
ignore_words = ['?', '!']

for intent in dataset['intents']:
    responses[intent['tag']] = intent['responses']
    for lines in intent['patterns']:
        inputs.append(lines)
        tags.append(intent['tag'])
        for pattern in intent['patterns']:
            w = nltk.word_tokenize(pattern)
            words.extend(w)
            documents.append((w, intent['tag']))
            if intent['tag'] not in classes:
                classes.append(intent['tag'])

data = pd.DataFrame({"patterns": inputs, "tags": tags})

data['patterns'] = data['patterns'].apply(lambda wrd: [ltrs.lower() for ltrs in wrd if ltrs not in string.punctuation])
data['patterns'] = data['patterns'].apply(lambda wrd: ''.join(wrd))

lemmatizer = WordNetLemmatizer()
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(data['patterns'])
train = tokenizer.texts_to_sequences(data['patterns'])

x_train = pad_sequences(train)

le = LabelEncoder()
y_train = le.fit_transform(data['tags'])

input_shape = x_train.shape[1]
vocabulary = len(tokenizer.word_index)
output_length = le.classes_.shape[0]

words = pickle.load(open('./pkl/words.pkl', 'rb'))
classes = pickle.load(open('./pkl/classes.pkl', 'rb'))
le = pickle.load(open('./pkl/le.pkl', 'rb'))
tokenizer = pickle.load(open('./pkl/tokenizers.pkl', 'rb'))

# Import the necessary libraries

# Load the model from the local file
model = tf.keras.models.load_model('./model/chatbot.h5', compile=False)

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print(f"found in bag: {w}")
    return(np.array(bag))


def predict_class(sentence):
    print("Input Message:", sentence)
    
    p = bow(sentence, words, show_details=False)
    # Trim or pad the input to match the expected length (49)
    p = p[:x_train]  # atau sesuaikan dengan logika preprocessing Anda
    p = np.array([p])  # Ubah menjadi array numpy
    
    res = model.predict(p)
    print("Predictions Shape:", res.shape)
    print("Raw Probabilities:", res[0])
    
    max_prob_index = np.argmax(res[0])  # Ambil indeks dengan probabilitas tertinggi
    
    return {"intent": classes[max_prob_index], "probability": float(res[0][max_prob_index])}

# Define the input shape based on your model's expected input shape
input_shape = x_train.shape[1]

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        data = request.get_json()
        message = data.get("message")
        
        # Call the model.predict function with the preprocessed message
        prediction_input = tokenizer.texts_to_sequences([message])
        prediction_input = np.array(prediction_input).reshape(1, -1)  # Adjust the reshaping
        sequence_length = x_train.shape[1]  # Use the desired sequence length here
        prediction_input = pad_sequences(prediction_input, maxlen=sequence_length)

        # Call the model.predict function with the preprocessed message
        output = model.predict(prediction_input)
        output = output.argmax()
        
        # Get the response tag and retrieve the actual response from the responses dictionary
        response_tag = le.inverse_transform([output])[0]
        response_text = random.choice(responses[response_tag])
        
        return jsonify({"response": {"response_text": response_text, "section": response_tag}})

if __name__ == "__main__":
    app.run(debug=True)