import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout
import re
from keras.callbacks import EarlyStopping

# Load the text data
with open(r'Z:\BrunoPersonalFiles\ChatbotDevelopment\Test3\Data\Raw\scraped_text_data.txt', 'r', encoding='utf-8') as f:
    text_data = f.read()

# Split the text into sentences or utterances
sentences = re.split(r'(?<=[.!?]) +', text_data)

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)
total_words = len(tokenizer.word_index) + 1

# Convert text data to sequences
input_sequences = tokenizer.texts_to_sequences(sentences)

# Pad sequences for uniform input size
max_sequence_len = max(len(seq) for seq in input_sequences)
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre')

# Create predictors and label
xs, labels = input_sequences[:, :-1], input_sequences[:, -1]
ys = np.array(labels)

# Define early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Define the model architecture
model = Sequential([
    
    Embedding(input_dim=total_words, output_dim=100, input_length=max_sequence_len-1),
    Bidirectional(LSTM(100,return_sequences=True)),
    Dropout(0.4),
    LSTM(100),
    Dense(total_words, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(xs, ys, epochs=100, callbacks=[early_stopping], verbose=1)

# Save the trained model
model.save("FinalModel")