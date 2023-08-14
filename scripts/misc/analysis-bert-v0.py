import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import numpy as np
from gensim import corpora, models
from gensim.similarities.termsim import WordEmbeddingSimilarityIndex
from gensim.similarities import SparseTermSimilarityMatrix
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)


def tokenize_and_preprocess(content):
    # You may need to improve this function according to your preprocessing requirements
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    import string

    stop_words = set(stopwords.words("english"))
    tokens = word_tokenize(content)
    cleaned_tokens = [
        word
        for word in tokens
        if word not in stop_words and word not in string.punctuation
    ]

    return cleaned_tokens


# Function to extract URL content
def extract_url_content(url, anchor_text):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    # Get main content
    main_content = soup.get_text()

    # Get specific section related to anchor text
    section_content = soup.find_all("p")
    sections = [section.get_text() for section in section_content]

    # Transform the documents into TF-IDF vectors
    tfidf = TfidfVectorizer().fit_transform([main_content] + sections)

    # Compute the cosine similarity between the TF-IDF vectors
    pairwise_similarity = tfidf * tfidf.T
    pairwise_similarity_array = pairwise_similarity.toarray()

    # Find the most similar document
    np.fill_diagonal(pairwise_similarity_array, np.nan)
    most_similar_section_idx = np.nanargmax(pairwise_similarity_array[0])
    most_similar_section = sections[most_similar_section_idx - 1]

    return main_content, most_similar_section


def extract_url_main(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    # Get main content
    main_content = soup.get_text()

    return main_content


# Function to compute BERT relevance
def compute_bert_relevance(page_content, link_content):
    embedder = SentenceTransformer(
        "bert-base-nli-mean-tokens"
    )  # Load pre-trained Sentence-BERT model

    # Convert sentences into embeddings
    page_embedding = embedder.encode([page_content])[0]
    link_embedding = embedder.encode([link_content])[0]

    # Compute cosine similarity between embeddings
    cosine_similarity = np.inner(page_embedding, link_embedding) / (
        np.linalg.norm(page_embedding) * np.linalg.norm(link_embedding)
    )

    return cosine_similarity


df = pd.read_csv("../data/dataset.tsv", sep="\t")
# Better display in the terminal
pd.set_option("expand_frame_repr", False)

# Apply functions
df[["link_main_content", "link_anchor_section"]] = df.apply(
    lambda row: pd.Series(extract_url_content(row["link"], row["anchor_text"])), axis=1
)
df["wikipedia_page_content"] = df["wikipedia_page"].apply(extract_url_main)
df["processed_wikipedia_content"] = df["wikipedia_page_content"].apply(
    tokenize_and_preprocess
)
df["processed_link_main_content"] = df["link_main_content"].apply(
    tokenize_and_preprocess
)
df["processed_link_anchor_section"] = df["link_anchor_section"].apply(
    tokenize_and_preprocess
)
df["processed_surrounding_paragraphs"] = df["surrounding_paragraph"].apply(
    tokenize_and_preprocess
)

# Similarity between full link content and paragraph surrounding anchor text
df["relevancy_bert_link_paragraph"] = df.apply(
    lambda row: compute_bert_relevance(
        " ".join(row["processed_surrounding_paragraphs"]),
        " ".join(row["processed_link_main_content"]),
    ),
    axis=1,
)
# Similarity between full link content and full wikipedia page
df["relevancy_bert_link_main"] = df.apply(
    lambda row: compute_bert_relevance(
        " ".join(row["processed_wikipedia_content"]),
        " ".join(row["processed_link_main_content"]),
    ),
    axis=1,
)

# Similarity between sections of the link content which are relevant to anchor text and paragraph surrounding the anchor text
df["relevancy_bert_anchor_paragraph"] = df.apply(
    lambda row: compute_bert_relevance(
        " ".join(row["processed_surrounding_paragraphs"]),
        " ".join(row["processed_link_anchor_section"]),
    ),
    axis=1,
)


print(df)
