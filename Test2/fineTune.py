# Load the initial model
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from keras.preprocessing.text import Tokenizer
import pickle
import numpy as np

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

def create_tokenizer(texts, vocab_size=1000, oov_token="<OOV>"):
    """
    Creates and fits a tokenizer on the given texts.
    
    Args:
    texts (list): A list of texts to fit the tokenizer on.
    vocab_size (int): The maximum size of the vocabulary.
    oov_token (str): Token to use for out-of-vocabulary words.
    
    Returns:
    keras.preprocessing.text.Tokenizer: The fitted tokenizer.
    """
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
    tokenizer.fit_on_texts(texts)
    return tokenizer

initial_model = load_model(r"Z:\BrunoPersonalFiles\ChatbotDevelopment\Test2\BaseModelFolder\chat_model")

# Prepare the labeled examples
labeled_data = []
with open('labeled_examples.txt', 'r') as file:
    for line in file:
        labeled_data.append(line.strip())

# Prepare the large text corpus
with open('large_text_corpus.txt', 'r') as file:
    large_text_corpus = file.read()

# Preprocess the labeled examples and large text corpus
preprocessed_labeled_data = [preprocess_text(example.split(',')[1]) for example in labeled_data]
preprocessed_large_text_corpus = preprocess_text(large_text_corpus)

# Combine the data
combined_data = preprocessed_labeled_data + [preprocessed_large_text_corpus]

# Create and fit tokenizer
tokenizer = create_tokenizer(combined_data)

# Tokenize and pad sequences
sequences_combined = tokenizer.texts_to_sequences(combined_data)
# Pad sequences with the correct length
padded_sequences_combined = pad_sequences(sequences_combined, maxlen=max_len, padding='post', truncating='post')

# Fine-tune the model
fine_tune_epochs = 60  # Adjust as needed
history_fine_tune = initial_model.fit(padded_sequences_combined, np.array([0]*len(preprocessed_labeled_data) + [1]), epochs=fine_tune_epochs)
initial_model.save(r"Z:\BrunoPersonalFiles\ChatbotDevelopment\Test2\FineTunedModelFolder\fine_tuned_model")