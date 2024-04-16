import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import re

def scrape_website(url, depth, scraped_urls=set()):
    if depth == 0 or url in scraped_urls:
        return ""

    print(f"Scraping {url}")
    scraped_urls.add(url)
    text_data = ""
    try:
        response = requests.get(url, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract headers (h1-h6) and paragraph text (p)
        headers = soup.find_all(re.compile(r'^h[1-6]$'))
        paragraphs = soup.find_all('p')

        # Extract text from headers and paragraphs
        header_texts = [header.get_text() for header in headers]
        paragraph_texts = [paragraph.get_text() for paragraph in paragraphs]

        # Combine header and paragraph text
        text_data += ' '.join(header_texts + paragraph_texts)

        # Find all links on the page and follow them up to depth - 1
        links = soup.find_all('a', href=True)
        for link in links:
            full_link = urljoin(url, link['href'])
            text_data += scrape_website(full_link, depth - 1, scraped_urls)
    except Exception as e:
        print(f"Failed to scrape {url}: {e}")
    return text_data


# List of starting URLs and depth
start_urls = [
    "https://dansrealenglish.com/conversation-topics-in-english/",
    "https://www.fluentu.com/blog/english/conversation-between-two-friends-in-english/",
    "https://promova.com/blog/conversations-in-english",
    "https://www.girlschase.com/article/conversation-example",
    "https://www.nationalgeographic.org/encyclopedia/ecology/#:~:text=Ecology%20is%20the%20study%20of%20organisms%20and%20how%20they%20interact,living%20things%20and%20their%20habitats.",
    "https://www.britannica.com/science/physics-science#:~:text=Physics%20is%20the%20branch%20of,entire%20universe%20using%20general%20relativity.",
    "https://www.tntech.edu/cas/physics/aboutphys/about-physics.php",
    "https://www.fluentu.com/blog/english/english-conversation-for-beginners/",
    "https://learnenglish.britishcouncil.org/skills/speaking",
    "https://www.eslfast.com/robot/",
    "https://realenglishconversations.com/",
    "https://agendaweb.org/listening/practical-english-conversations.html",
    "https://www.bbc.co.uk/learningenglish/english/features/the-english-we-speak"
    
]
depth = 3

# Scrape websites and get raw text
raw_text_data = ""
for url in start_urls:
    raw_text_data += scrape_website(url, depth)

# Save raw text to a text file
with open(r'Z:\BrunoPersonalFiles\ChatbotDevelopment\Test3\Data\Raw\scraped_text_data.txt', 'w', encoding='utf-8') as file:
    file.write(raw_text_data)

print("Scraped text data saved to scraped_text_data.txt")
