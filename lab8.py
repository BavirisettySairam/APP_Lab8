import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
import os
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize, PunktSentenceTokenizer
from nltk.stem import PorterStemmer
import time
import urllib.request

# Set a temporary directory for NLTK data (works in Streamlit Cloud)
NLTK_PATH = "/tmp/nltk_data"

# Ensure the directory exists
os.makedirs(NLTK_PATH, exist_ok=True)
os.makedirs(os.path.join(NLTK_PATH, 'tokenizers/punkt'), exist_ok=True)

# Append the path to nltk data
nltk.data.path.append(NLTK_PATH)

# Download english.pickle directly
punkt_url = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/tokenizers/punkt/english.pickle"
punkt_path = os.path.join(NLTK_PATH, 'tokenizers/punkt/english.pickle')

try:
    if not os.path.exists(punkt_path):
        print(f"Downloading english.pickle from: {punkt_url}")
        urllib.request.urlretrieve(punkt_url, punkt_path)
        print(f"english.pickle downloaded to: {punkt_path}")
    else:
        print(f"english.pickle already exists at: {punkt_path}")

    # Load the Punkt Sentence Tokenizer
    sent_tokenizer = PunktSentenceTokenizer(punkt_path)
    print("Punkt Sentence Tokenizer loaded successfully.")

except Exception as e:
    print(f"Error loading Punkt Sentence Tokenizer: {e}")

try:
    nltk.data.find('stopwords')
    print("stopwords already downloaded.")
except LookupError:
    print("Downloading stopwords...")
    nltk.download('stopwords', download_dir=NLTK_PATH)
    print("stopwords downloaded successfully.")
    time.sleep(1)

# Text Preprocessing function
def preprocess_text(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in nltk.corpus.stopwords.words('english')]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

# ... rest of your code ...

# ... rest of your code ...
# Streamlit Page Configuration
st.set_page_config(page_title="Lab 8", page_icon="📊", layout="centered")

# Title and Author Info
st.title("📊 Lab 8 - Review Analysis")
st.markdown("### Made by **Bavirisetty Sairam - 2447115**")

# Define file paths
FILE_MAIN = "Restaurant_Reviews.tsv"
FILE_TEXT1 = "text1.tsv"
FILE_TEXT2 = "text2.tsv"

# Text Preprocessing function
# def preprocess_text(text):
#     if not isinstance(text, str) or not text.strip():
#         return ""
#     text = text.lower()
#     text = re.sub(r'[^a-zA-Z\s]', '', text)
#     tokens = word_tokenize(text)
#     tokens = [word for word in tokens if word not in nltk.corpus.stopwords.words('english')]
#     stemmer = PorterStemmer()
#     tokens = [stemmer.stem(word) for word in tokens]
#     return ' '.join(tokens)

# Load and preprocess datasets
@st.cache_data
def load_and_process_data():
    df_main = pd.read_csv(FILE_MAIN, delimiter='\t')
    df_text1 = pd.read_csv(FILE_TEXT1, delimiter='\t')
    df_text2 = pd.read_csv(FILE_TEXT2, delimiter='\t')
    
    df_main['processed_review'] = df_main['Review'].astype(str).apply(preprocess_text)
    df_text1['processed_review'] = df_text1['Review'].astype(str).apply(preprocess_text)
    df_text2['processed_review'] = df_text2['Review'].astype(str).apply(preprocess_text)
    
    return df_main, df_text1, df_text2

df_main, df_text1, df_text2 = load_and_process_data()

# Sidebar Options
st.sidebar.title("🔧 Options")
show_data = st.sidebar.checkbox("Show Original Data")
show_wordcloud = st.sidebar.checkbox("Show Word Cloud")
show_similarity = st.sidebar.checkbox("Show Similarity Analysis")

if show_data:
    st.subheader("🔍 Data Preview")
    st.write(df_main[['Review', 'processed_review']].head())

# Generate and show Word Cloud
if show_wordcloud:
    st.subheader("🌥️ Word Cloud - Most Common Words")
    fig, ax = plt.subplots(figsize=(10, 5))
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df_main['processed_review']))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)

if show_similarity:
    # Convert to TF-IDF for similarity calculations
    vectorizer = TfidfVectorizer()
    tfidf_matrix_text1 = vectorizer.fit_transform(df_text1['processed_review'])
    tfidf_matrix_text2 = vectorizer.transform(df_text2['processed_review'])

    # Compute Cosine Similarity
    cos_sim = cosine_similarity(tfidf_matrix_text1, tfidf_matrix_text2)

    st.subheader("📏 Cosine Similarity Scores")
    st.write(pd.DataFrame(cos_sim, index=[f"T1-{i+1}" for i in range(len(df_text1))], 
                            columns=[f"T2-{i+1}" for i in range(len(df_text2))]))

    # Compute Jaccard Similarity
    def jaccard_similarity(set1, set2):
        return len(set1.intersection(set2)) / len(set1.union(set2))

    jaccard_scores = np.zeros((len(df_text1), len(df_text2)))
    for i in range(len(df_text1)):
        for j in range(len(df_text2)):
            set1, set2 = set(df_text1['processed_review'][i].split()), set(df_text2['processed_review'][j].split())
            jaccard_scores[i][j] = jaccard_similarity(set1, set2)

    # Improved Jaccard Similarity Heatmap
    st.subheader("🔥 Jaccard Similarity Heatmap")
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(jaccard_scores, cmap='coolwarm', annot=True, fmt=".2f",
                xticklabels=[f"T2-{i+1}" for i in range(len(df_text2))], 
                yticklabels=[f"T1-{i+1}" for i in range(len(df_text1))],
                annot_kws={"size": 10})

    plt.xticks(rotation=30, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.title("Jaccard Similarity Heatmap", fontsize=14)

    st.pyplot(fig)

st.success("✅ Analysis Complete!")
