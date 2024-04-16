import requests
from bs4 import BeautifulSoup
import re
import nltk
from urllib.parse import urljoin
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Ensure you've downloaded the necessary NLTK resources
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('stopwords')

def scrape_website(url, depth, scraped_urls=set()):
    if depth == 0 or url in scraped_urls:
        return ""

    print(f"Scraping {url}")
    scraped_urls.add(url)
    text_data = ""
    try:
        response = requests.get(url, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Process the text of the current page
        paragraphs = soup.find_all('p')
        text_data += ' '.join([p.get_text() for p in paragraphs])

        # Find all links on the page and follow them up to depth - 1
        links = soup.find_all('a', href=True)
        for link in links:
            full_link = urljoin(url, link['href'])
            text_data += scrape_website(full_link, depth - 1, scraped_urls)
    except Exception as e:
        print(f"Failed to scrape {url}: {e}")
    return text_data

def preprocess_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text).lower()
    tokens = word_tokenize(text)
    stop_words = set(nltk.corpus.stopwords.words('english'))
    tokens = [word for word in tokens if not word in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Starting URLs
urls = [
    'https://www.tensorflow.org/tutorials',
    # Add more URLs as needed
]
depth = 2

# Combine text from all URLs and preprocess
combined_text = ""
for url in urls:
    combined_text += scrape_website(url, depth)  # Adjust depth as needed

preprocessed_data = preprocess_text(combined_text)

# Write preprocessed data to a text file
with open('large_text_corpus.txt', 'w') as file:
    file.write(preprocessed_data)

print("Preprocessed data saved to preprocessed_data.txt")
