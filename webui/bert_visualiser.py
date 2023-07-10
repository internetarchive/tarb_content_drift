import time
from sentence_transformers import SentenceTransformer
import numpy as np
import requests
from bs4 import BeautifulSoup
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
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


st.set_page_config(layout="wide")

st.title("BERT Relevance Calculator")

webpage = st.text_input("Enter webpage URL", "")
context_string = st.text_input("Enter context string", "")

if st.button("Calculate Relevance"):
    if webpage and context_string:
        start_time = time.time()
        relevance, webpage_embedding, string_embedding = compute_bert_relevance(
            webpage, context_string
        )
        end_time = time.time()
        runtime = end_time - start_time
        col1, col2, col3 = st.columns(3)
        runtime = (end_time - start_time) * 1000

        with col1:
            st.subheader("Relevance Score:")
            st.markdown(
                f"""
            <div style="font-size:24px; color:lightgreen">
                {relevance * 100:.1f}%
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col2:
            st.subheader("Runtime:")
            st.markdown(
                f"""
            <div style="font-size:24px; color:pink">
                {runtime:.2f} ms
            </div>
            """,
                unsafe_allow_html=True,
            )

        with st.sidebar:
            st.header("How does this work?")
            st.markdown(
                """
            This application uses the BERT (Bidirectional Encoder Representations from Transformers) model to measure the relevance between the text content of a provided webpage URL and a context string.
            
            1. **Web Content Extraction**: The script fetches and parses the provided webpage using Python's BeautifulSoup library to extract all text.

            2. **Preprocessing**: The extracted text and the context string are tokenized and cleaned of stopwords and punctuation.

            3. **BERT Embedding**: Each tokenized and cleaned text is sent through the BERT model to produce contextually-rich embeddings.

            4. **Relevance Score Calculation**: The cosine similarity between the embeddings is calculated to yield a relevance score. The closer the score is to 1 (or 100%), the more relevant the webpage content is to the context string.
            
            5. **Visualization**: The 768-dimensional BERT embeddings for the webpage and the context string are reduced to 2 dimensions using PCA for easy visualization.
            """
            )

            embed_col1, embed_col2 = st.columns(2)
            with embed_col1:
                st.markdown(
                    "<h4 style='font-size:12px;'>Webpage Embedding:</h4>",
                    unsafe_allow_html=True,
                )
                st.write(webpage_embedding)
            with embed_col2:
                st.markdown(
                    "<h4 style='font-size:12px;'>Context String Embedding:</h4>",
                    unsafe_allow_html=True,
                )
                st.write(string_embedding)

            embeddings = np.stack([webpage_embedding, string_embedding])
            pca = PCA(n_components=2)
            embeddings_2d = pca.fit_transform(embeddings)

            st.write("Here's a plot to visualize how the embeddings are compared:")
            fig, ax = plt.subplots()
            ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], color="blue")
            ax.plot(embeddings_2d[:, 0], embeddings_2d[:, 1], color="red")

            for i, txt in enumerate(["Webpage Embedding", "Context String Embedding"]):
                ax.annotate(
                    txt,
                    (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                    textcoords="offset points",
                    xytext=(10, 10),
                    ha="center",
                )

            ax.axhline(0, color="black", linewidth=0.5)
            ax.axvline(0, color="black", linewidth=0.5)
            ax.grid(color="gray", linestyle="--", linewidth=0.5)
            st.pyplot(fig)

    else:
        st.warning("Please enter both webpage URL and context string.")
