import requests
import re
from SimpleTokenizerV1 import SimpleTokenizerV1

def generate_vocab(text):
    preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    all_words = sorted(list(set(preprocessed)))
    vocab = {token: integer for integer, token in enumerate(all_words)}
    return vocab

# get data
url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
response = requests.get(url)
text = response.text
# init tokenizer
tokenizer = SimpleTokenizerV1(generate_vocab(text))

#test tokenizer
text_shorter = """"It's the last he painted, you know," Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text_shorter)
decoded = tokenizer.decode(ids)

print(ids)
print(decoded)
