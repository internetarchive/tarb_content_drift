import requests
from bs4 import BeautifulSoup
import numpy as np
from gensim import corpora, models
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer
import multiprocessing as mp
import nltk
from nltk.corpus import stopwords
import os
import certifi

os.environ["SSL_CERT_FILE"] = certifi.where()

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)


def fetch_wikipedia_page(url):
    response = requests.get(url)
    content = response.text
    return content


def extract_links_and_anchor_texts(content):
    soup = BeautifulSoup(content, "html.parser")
    internal_links = []
    external_links = []
    internal_anchor_texts = []
    external_anchor_texts = []

    for link in soup.find_all("a", href=True):
        if link["href"].startswith("/wiki/") and ":" not in link["href"]:
            internal_links.append("https://en.wikipedia.org" + link["href"])
            internal_anchor_texts.append(link.text)
        elif not link["href"].startswith("#") and not link["href"].startswith("/"):
            external_links.append(link["href"])
            external_anchor_texts.append(link.text)

    return internal_links, internal_anchor_texts, external_links, external_anchor_texts


def preprocess(text):
    result = []
    nltk_stopwords = set(stopwords.words("english"))
    for token in simple_preprocess(text):
        if token not in nltk_stopwords and len(token) > 3:
            result.append(WordNetLemmatizer().lemmatize(token, pos="v"))
    return result


def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)


def process_window(idx, words, anchor_text, window_size):
    window_content = " ".join(words[idx : idx + window_size])
    preprocessed_texts = [preprocess(anchor_text), preprocess(window_content)]

    dictionary = corpora.Dictionary(preprocessed_texts)
    corpus = [dictionary.doc2bow(text) for text in preprocessed_texts]

    lda_model = models.LdaModel(corpus, num_topics=10, id2word=dictionary, passes=15)

    anchor_text_topic_distribution = lda_model.get_document_topics(
        corpus[0], minimum_probability=0
    )
    window_content_topic_distribution = lda_model.get_document_topics(
        corpus[1], minimum_probability=0
    )

    similarity = cosine_similarity(
        [prob for _, prob in anchor_text_topic_distribution],
        [prob for _, prob in window_content_topic_distribution],
    )

    return (similarity, window_content)


def find_closest_content_lda(main_page_text, anchor_text, window_size=100):
    words = main_page_text.split()
    best_similarity = 0
    best_content = ""

    with mp.Pool() as pool:
        results = pool.starmap(
            process_window,
            [
                (idx, words, anchor_text, window_size)
                for idx in range(len(words) - window_size + 1)
            ],
        )

    for similarity, window_content in results:
        if similarity > best_similarity:
            best_similarity = similarity
            best_content = window_content

    return best_content, best_similarity


def analyze_links(main_page_text, links, anchor_texts, link_type, num_links_to_analyze):
    for idx, link in enumerate(links[:num_links_to_analyze]):
        anchor_text = anchor_texts[idx]
        closest_content, scoped_relevancy = find_closest_content_lda(
            main_page_text, anchor_text, window_size=200
        )

        preprocessed_texts = [preprocess(main_page_text), preprocess(anchor_text)]
        dictionary = corpora.Dictionary(preprocessed_texts)
        corpus = [dictionary.doc2bow(text) for text in preprocessed_texts]

        lda_model = models.LdaModel(
            corpus, num_topics=10, id2word=dictionary, passes=15
        )

        main_page_topic_distribution = lda_model.get_document_topics(
            corpus[0], minimum_probability=0
        )
        anchor_text_topic_distribution = lda_model.get_document_topics(
            corpus[1], minimum_probability=0
        )

        overall_relevancy = cosine_similarity(
            [prob for _, prob in main_page_topic_distribution],
            [prob for _, prob in anchor_text_topic_distribution],
        )

        print(f"Link: {link}")
        print(f"Anchor text: {anchor_text}")
        print(f"Overall relevancy score: {overall_relevancy * 100:.2f}%")
        print(f"Scoped relevancy score: {scoped_relevancy * 100:.2f}%")
        print(f"Similar content: {closest_content}")
        print()


def get_test_links():
    with open("tests/links.txt", "r") as file:
        links = file.readlines()
    return [link.strip() for link in links]


def main():
    test_links = get_test_links()
    for url in test_links:
        main_page_content = fetch_wikipedia_page(url)
        main_page_text = BeautifulSoup(main_page_content, "html.parser").get_text()
        (
            internal_links,
            internal_anchor_texts,
            external_links,
            external_anchor_texts,
        ) = extract_links_and_anchor_texts(main_page_content)

        analyze_links(
            main_page_text,
            external_links,
            external_anchor_texts,
            "external",
            num_links_to_analyze=10,
        )


if __name__ == "__main__":
    main()
    
