import requests
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Function to scrape text data from a website
def scrape_website(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    # Extract text from all paragraphs
    paragraphs = soup.find_all('p')
    text = ' '.join([p.get_text() for p in paragraphs])
    return text

# Function to preprocess text data
def preprocess_text(text):
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove non-alphanumeric characters and convert to lowercase
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text).lower()
    # Tokenization
    tokens = word_tokenize(text)
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # Join tokens back into a string
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

# URL of the website to scrape
url = 'https://flask.palletsprojects.com/en/3.0.x/'

# Scrape text data from the website
text_data = scrape_website(url)

# Preprocess the text data
preprocessed_data = preprocess_text(text_data)

# Write preprocessed data to a text file
with open('preprocessed_data.txt', 'w') as file:
    file.write(preprocessed_data)

print("Preprocessed data saved to preprocessed_data.txt")
