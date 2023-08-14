import requests
from bs4 import BeautifulSoup
import pandas as pd

# Dataframe with the urls of the embedded links


def extract_data(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    data = []

    paragraphs = soup.find_all("p")
    for paragraph in paragraphs:
        a_tags = paragraph.find_all("a", href=True)
        for a in a_tags:
            row = {}
            href = a["href"]
            if href.startswith("/") or href.startswith("#"):
                href = url + href
            row["link"] = href
            row["anchor_text"] = a.text
            row["paragraph_heading"] = (
                paragraph.find_previous("h2").text
                if paragraph.find_previous("h2")
                else None
            )
            row["paragraph_content"] = paragraph.text
            data.append(row)

    df = pd.DataFrame(data)
    return df


# Current rows : link, anchor_text, paragraph_heading, paragraph_content


# Function to derive two contexts from the wikipedia page and the url content


def main():
    print(extract_data("https://en.wikipedia.org/wiki/Artificial_intelligence"))
