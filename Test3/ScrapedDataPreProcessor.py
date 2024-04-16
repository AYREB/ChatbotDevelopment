import re
import json

def custom_tokenizer(text):
    # Split text on spaces
    tokens = text.split()
    
    # Merge tokens containing an apostrophe
    merged_tokens = []
    current_token = ''
    for token in tokens:
        if "'" in token:
            current_token += token
        else:
            if current_token:
                merged_tokens.append(current_token)
                current_token = ''
            merged_tokens.append(token)
    if current_token:
        merged_tokens.append(current_token)
    
    # Split punctuation marks into their own tokens
    final_tokens = []
    for token in merged_tokens:
        final_tokens.extend(re.split(r'([.,])', token))
    
    # Remove empty strings from the list
    final_tokens = [token for token in final_tokens if token]
    
    return final_tokens

def process_text_file(input_file, output_file):
    # Read text from input file
    with open(input_file, 'r', encoding='utf-8') as file:
        text = file.read()
    
    # Remove line breaks
    text = text.replace('\n', ' ')
    
    # Tokenize the text
    tokens = custom_tokenizer(text)
    
    # Create a list of dictionaries where each dictionary contains the tokenized words
    tokenized_sentences = [{'words': tokens}]
    
    # Save tokenized sentences to output JSON file
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(tokenized_sentences, file, indent=4, ensure_ascii=False)

# Path to input and output files
input_file = r'Z:\BrunoPersonalFiles\ChatbotDevelopment\Test3\Data\Raw\scraped_text_data.txt'
output_file = r'Z:\BrunoPersonalFiles\ChatbotDevelopment\Test3\Data\Tokenised\processed_data.json'

# Process the text file
process_text_file(input_file, output_file)

print("Tokens saved to:", output_file)