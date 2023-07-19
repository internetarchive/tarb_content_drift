from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import TruncatedSVD
import requests
from bs4 import BeautifulSoup


def get_only_text(url):
    """
    return the title and the text of the article
    at the specified url
    """
    page = requests.get(url)
    soup = BeautifulSoup(page.text, "html.parser")
    text = " ".join(map(lambda p: p.text, soup.find_all("p")))
    return soup.title.text, text


url = "https://en.wikipedia.org/wiki/BERT_(language_model)"
title, text = get_only_text(url)

# Load BERT model
model = SentenceTransformer("bert-base-uncased")

# Define your context string
context_string = """"""

# Generate BERT embeddings for both texts
text_embeddings = model.encode([text], convert_to_tensor=True)
context_embeddings = model.encode([context_string], convert_to_tensor=True)

# Convert embeddings to numpy array and reshape to 2D
text_embeddings = text_embeddings.detach().numpy().reshape(1, -1)
context_embeddings = context_embeddings.detach().numpy().reshape(1, -1)

# Calculate cosine similarity
cos_similarity = cosine_similarity(text_embeddings, context_embeddings)

print("Cosine Similarity: ", cos_similarity)
