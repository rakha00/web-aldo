import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import LabelEncoder
import nltk
import sys

# Add parent directory to sys.path to import preprocessing.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'streamlit_app')))
from preprocessing import text_length_features # Import the function

# Download NLTK data if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')

# Load data
data = pd.read_csv('content/hasil_labeling.csv')

# Label Encoding
le = LabelEncoder()
data['label_encoded'] = le.fit_transform(data['sentiment'])

# Data Splitting (using 80:20 split as an example for transformer fitting)
X = data['stemming']
y = data['label_encoded']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# TF-IDF Word-level
tfidf_word = TfidfVectorizer(
    analyzer='word',
    ngram_range=(1,2),
    max_features=10000
)
tfidf_word.fit(X_train)

# TF-IDF Char-level
tfidf_char = TfidfVectorizer(
    analyzer='char_wb',
    ngram_range=(2,5),
    max_features=10000
)
tfidf_char.fit(X_train)

# Text Length Features
length_transformer = FunctionTransformer(text_length_features, validate=False)
length_transformer.fit(X_train) # Fit the transformer on X_train

# Create models directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

# Save the transformers
joblib.dump(tfidf_word, 'models/tfidf_word_vectorizer.pkl')
joblib.dump(tfidf_char, 'models/tfidf_char_vectorizer.pkl')
joblib.dump(length_transformer, 'models/length_transformer.pkl')

print("Transformers saved successfully to the 'models' directory.")