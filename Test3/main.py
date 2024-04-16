import json
import numpy as np
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
import keras
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense

# Load the JSON data
with open(r'Z:\BrunoPersonalFiles\ChatbotDevelopment\Test3\Data\Tokenised\processed_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Extract topics
topics = [item['words'] for item in data]

# Tokenize the topics
tokenizer = Tokenizer()
tokenizer.fit_on_texts(topics)
total_words = len(tokenizer.word_index) + 1

# Convert text to sequences
input_sequences = tokenizer.texts_to_sequences(topics)

# Pad sequences for uniform input size
max_sequence_len = max(len(seq) for seq in input_sequences)
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre')

# Create predictors and label
xs, labels = input_sequences[:, :-1], input_sequences[:, -1]
ys = np.array(labels)

# Step 2: Model Definition
model = Sequential([
    Embedding(input_dim=total_words, output_dim=100, input_length=max_sequence_len - 1),
    LSTM(150),
    Dense(total_words, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Step 3: Model Training
history = model.fit(xs, ys, epochs=100, verbose=1)

# Step 4: Model Evaluation (Optional)
# Evaluate the model on the training data
loss, accuracy = model.evaluate(xs, ys, verbose=0)
print(f'Training Loss: {loss:.4f}, Training Accuracy: {accuracy:.4f}')

model.save("FinalModel")