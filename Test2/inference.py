import numpy as np
from tensorflow import keras
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import json
import pickle
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import colorama
colorama.init()
from colorama import Fore, Style

# Load the fine-tuned model
model = keras.models.load_model(r'Z:\BrunoPersonalFiles\ChatbotDevelopment\Test2\FineTunedModelFolder\fine_tuned_model')

# Load the tokenizer object
with open(r'Z:\BrunoPersonalFiles\ChatbotDevelopment\Test2\BaseModelFolder\tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load the label encoder object
with open(r'Z:\BrunoPersonalFiles\ChatbotDevelopment\Test2\BaseModelFolder\label_encoder.pickle', 'rb') as enc:
    lbl_encoder = pickle.load(enc)

max_len = 20

def preprocess_text(text):
    """
    Preprocesses the given text by removing HTML tags, non-alphanumeric characters,
    converting to lowercase, tokenizing, removing stop words, and lemmatizing.
    
    Args:
    text (str): The text to preprocess.
    
    Returns:
    str: The preprocessed text.
    """
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text).lower()  # Remove non-alphanumeric characters and convert to lowercase
    tokens = word_tokenize(text)  # Tokenize the text
    stop_words = set(nltk.corpus.stopwords.words('english'))
    tokens = [word for word in tokens if not word in stop_words]  # Remove stop words
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatize words
    return ' '.join(tokens)

def chat():
    while True:
        print(Fore.LIGHTBLUE_EX + "User: " + Style.RESET_ALL, end="")
        inp = input()
        if inp.lower() == "quit":
            break
        
        # Preprocess input
        preprocessed_input = preprocess_text(inp)
        
        # Tokenize and pad sequence
        input_sequence = tokenizer.texts_to_sequences([preprocessed_input])
        padded_input_sequence = pad_sequences(input_sequence, maxlen=max_len, padding='post', truncating='post')
        
        # Get model prediction
        result = model.predict(padded_input_sequence)
        tag = lbl_encoder.inverse_transform([np.argmax(result)])[0]

        # Find appropriate response based on predicted tag
        for intent in intents['intents']:
            if intent['tag'] == tag:
                print(Fore.GREEN + "ChatBot:" + Style.RESET_ALL , np.random.choice(intent['responses']))

# Load intents from file
with open(r"Z:\BrunoPersonalFiles\ChatbotDevelopment\Test2\intents.json") as file:
    intents = json.load(file)

print(Fore.YELLOW + "Start messaging with the bot (type quit to stop)!" + Style.RESET_ALL)
chat()
