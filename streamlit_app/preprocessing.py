import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import numpy as np # Add numpy import

# Ensure NLTK data is downloaded
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Initialize Sastrawi Stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Load stop words
list_stopwords = set(stopwords.words('indonesian'))

def clean_text(text):
    text = re.sub(r'@[A-Za-z0-9_]+', '', text) # Remove mentions
    text = re.sub(r'#\w+', '', text) # Remove hashtags
    text = re.sub(r'RT[\s]+', '', text) # Remove RT
    text = re.sub(r'https?://\S+|www\.\S+', '', text) # Remove URLs
    text = re.sub(r'\d+', '', text) # Remove numbers
    text = re.sub(r'\b[a-zA-Z]\b', '', text) # Remove single characters
    text = re.sub(r'\s+', ' ', text).strip() # Remove extra spaces
    return text

def case_folding(text):
    return text.lower()

def remove_non_alphanumeric(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

def remove_emojis(text):
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE
    )
    return emoji_pattern.sub(r'', text)

def remove_stopwords_text(text):
    word_tokens = word_tokenize(text)
    filtered_sentence = [w for w in word_tokens if not w in list_stopwords]
    return " ".join(filtered_sentence)

def stem_text(text):
    return stemmer.stem(text)

def preprocess_text(text):
    text = clean_text(text)
    text = case_folding(text)
    text = remove_non_alphanumeric(text)
    text = remove_emojis(text)
    text = remove_stopwords_text(text)
    text = stem_text(text)
    return text

# Define text_length_features here
def text_length_features(texts):
    char_lengths = np.array([len(text) for text in texts]).reshape(-1, 1)
    word_lengths = np.array([len(text.split()) for text in texts]).reshape(-1, 1)
    return np.hstack((char_lengths, word_lengths))