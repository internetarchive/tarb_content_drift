import requests
from bs4 import BeautifulSoup
from readability import Document

# HTTP Session
session = requests.Session()


def is_page_unavailable(content):
    unavailable_phrases = [
        "This Page is Unavailable Right Now",
        "404 Not Found",
        "Page Not Found",
    ]
    return any(phrase in content for phrase in unavailable_phrases)


def get_relevant_content(url, anchor_text):
    response = session.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    anchor_tag = soup.find("a", string=anchor_text)
    if anchor_tag:
        relevant_container = anchor_tag.find_parent(["p", "div", "li"])
        return relevant_container.get_text() if relevant_container else ""
    return ""


def get_embedded_content(url):
    response = session.get(url)
    doc = Document(response.text)
    summary_html = doc.summary()
    soup = BeautifulSoup(summary_html, "html.parser")
    return soup.get_text()
