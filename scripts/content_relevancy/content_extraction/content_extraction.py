import requests
from bs4 import BeautifulSoup
from readability import Document

# Define headers to include User-Agent
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}


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

        if relevant_container:
            content_text = relevant_container.get_text()
            word_count = len(content_text.split())

            prev_sibling = relevant_container.find_previous_sibling(["p", "div", "li"])
            next_sibling = relevant_container.find_next_sibling(["p", "div", "li"])

            while word_count < 300:
                if prev_sibling is not None:
                    content_text = prev_sibling.get_text() + " " + content_text
                    word_count = len(content_text.split())
                    prev_sibling = prev_sibling.find_previous_sibling(
                        ["p", "div", "li"]
                    )

                if word_count >= 300:
                    break

                if next_sibling is not None:
                    content_text += " " + next_sibling.get_text()
                    word_count = len(content_text.split())
                    next_sibling = next_sibling.find_next_sibling(["p", "div", "li"])

                if prev_sibling is None and next_sibling is None:
                    break

            return content_text
    return ""


def get_embedded_content(url):
    response = session.get(url)
    doc = Document(response.text)
    summary_html = doc.summary()
    soup = BeautifulSoup(summary_html, "html.parser")
    return soup.get_text()
