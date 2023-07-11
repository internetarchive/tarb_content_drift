import numpy as np
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import nltk
import os
import certifi

os.environ["SSL_CERT_FILE"] = certifi.where()

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)


def tokenize_and_preprocess(content):
    stop_words = set(stopwords.words("english"))
    tokens = word_tokenize(content)
    cleaned_tokens = [
        word
        for word in tokens
        if word not in stop_words and word not in string.punctuation
    ]
    return cleaned_tokens


def compute_bert_relevance(webpage, context_string):
    page = requests.get(webpage)
    soup = BeautifulSoup(page.content, "html.parser")

    webpage_text = "".join(node.get_text() for node in soup.find_all())
    webpage_tokens = tokenize_and_preprocess(webpage_text)
    context_string_tokens = tokenize_and_preprocess(context_string)

    embedder = SentenceTransformer("bert-base-nli-mean-tokens")

    webpage_embedding = embedder.encode([" ".join(webpage_tokens)])[0]
    string_embedding = embedder.encode([" ".join(context_string_tokens)])[0]

    cosine_similarity = np.inner(webpage_embedding, string_embedding) / (
        np.linalg.norm(webpage_embedding) * np.linalg.norm(string_embedding)
    )

    return cosine_similarity, webpage_embedding, string_embedding
