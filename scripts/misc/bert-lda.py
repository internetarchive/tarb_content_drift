import requests
from bs4 import BeautifulSoup
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from umap import UMAP
import numpy as np

# Load BERT model
model = SentenceTransformer("bert-base-uncased")

# Define the URL of the web page
url = "https://en.wikipedia.org/wiki/BERT_(language_model)"

# Make a request to the website
r = requests.get(url)
# Use BeautifulSoup to parse the HTML content of the page
soup = BeautifulSoup(r.text, "html.parser")

# Extract the text content of the web page
web_page_content = soup.get_text()

context_string = """BERT (Bidirectional Encoder Representations from Transformers) is a recent paper published by researchers at Google AI Language. It has caused a stir in the Machine Learning community by presenting state-of-the-art results in a wide variety of NLP tasks, including Question Answering (SQuAD v1.1), Natural Language Inference (MNLI), and others."""

# Define your documents
documents = [web_page_content, context_string]

# Generate BERT embeddings
embeddings = model.encode(documents, convert_to_tensor=True)

# Convert embeddings to numpy array for UMAP
embeddings = embeddings.detach().numpy()

# Check if the embeddings are empty
if embeddings.size == 0:
    raise ValueError("Your embeddings are empty. Double check your input data.")
else:
    # Dimensionality reduction using UMAP with customized parameters
    reducer = UMAP(n_neighbors=15, min_dist=0.1)
    reduced_embeddings = reducer.fit_transform(embeddings)

    # Clustering
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(reduced_embeddings)

    # Silhouette Score
    sil_score = silhouette_score(reduced_embeddings, kmeans.labels_)

    print("Silhouette Score: ", sil_score)
