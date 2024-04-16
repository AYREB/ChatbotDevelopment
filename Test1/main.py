
import json 
import numpy as np 
import tensorflow as tf
import keras
import keras
from keras.models import Sequential
from keras.layers import Dense, Embedding, GlobalAveragePooling1D, BatchNormalization, Dropout
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import pickle


with open(r'Z:\BrunoPersonalFiles\ChatbotDevelopment\Test1\intents.json') as file:
    data = json.load(file)
    
training_sentences = []
training_labels = []
labels = []
responses = []


for intent in data['intents']:
    for pattern in intent['patterns']:
        training_sentences.append(pattern)
        training_labels.append(intent['tag'])
    responses.append(intent['responses'])
    
    if intent['tag'] not in labels:
        labels.append(intent['tag'])
        
num_classes = len(labels)


lbl_encoder = LabelEncoder()
lbl_encoder.fit(training_labels)
training_labels = lbl_encoder.transform(training_labels)



vocab_size = 1000
embedding_dim = 16
max_len = 20
oov_token = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(training_sentences)
padded_sequences = pad_sequences(sequences, truncating='post', maxlen=max_len)


# Create a Sequential model
model = Sequential()

# Add an embedding layer
model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))

# Add global average pooling layer
model.add(GlobalAveragePooling1D())

# Add a dense layer with ReLU activation and dropout
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))

# Add batch normalization layer
model.add(BatchNormalization())

# Add another dense layer with ReLU activation and dropout
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))

# Output layer with softmax activation
model.add(Dense(num_classes, activation='softmax'))

# Compile the model with a different optimizer
optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Print a summary of the model
model.summary()

epochs = 500
history = model.fit(padded_sequences, np.array(training_labels), epochs=epochs)


# to save the trained model
model.save("chat_model")



# to save the fitted tokenizer
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
# to save the fitted label encoder
with open('label_encoder.pickle', 'wb') as ecn_file:
    pickle.dump(lbl_encoder, ecn_file, protocol=pickle.HIGHEST_PROTOCOL)