import requests
from bs4 import BeautifulSoup
import re
from gensim import corpora, models
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
import nltk
from nltk.stem import WordNetLemmatizer
import os
import certifi

os.environ["SSL_CERT_FILE"] = certifi.where()

nltk.download("wordnet")


def fetch_wikipedia_page(url):
    response = requests.get(url)
    content = response.text
    return content


def extract_links_and_anchor_texts(content):
    soup = BeautifulSoup(content, "html.parser")
    links = []
    anchor_texts = []

    for link in soup.find_all("a", href=True):
        if link["href"].startswith("/wiki/") and ":" not in link["href"]:
            links.append("https://en.wikipedia.org" + link["href"])
            anchor_texts.append(link.text)

    return links, anchor_texts


def preprocess(text):
    result = []
    for token in simple_preprocess(text):
        if token not in STOPWORDS and len(token) > 3:
            result.append(WordNetLemmatizer().lemmatize(token, pos="v"))
    return result


def get_test_links():
    with open("../tests/links.txt", "r") as file:
        links = file.readlines()
    return [link.strip() for link in links]


def main():
    test_links = get_test_links()
    for url in test_links:
        main_page_content = fetch_wikipedia_page(url)
        main_page_text = BeautifulSoup(main_page_content, "html.parser").get_text()
        links, anchor_texts = extract_links_and_anchor_texts(main_page_content)

        preprocessed_texts = [preprocess(main_page_text)] + [
            preprocess(text) for text in anchor_texts
        ]

        dictionary = corpora.Dictionary(preprocessed_texts)
        corpus = [dictionary.doc2bow(text) for text in preprocessed_texts]

        lda_model = models.LdaModel(
            corpus, num_topics=10, id2word=dictionary, passes=15
        )

        for idx, link in enumerate(links):
            print(f"Link: {link}")
            print("Most common topics:")
            topics = lda_model.get_document_topics(
                corpus[idx + 1], minimum_probability=0.1
            )
            for topic_id, prob in topics:
                print(
                    f"Topic {topic_id}: {lda_model.print_topic(topic_id, 5)} (probability: {prob:.2f})"
                )
            print()


if __name__ == "__main__":
    main()
