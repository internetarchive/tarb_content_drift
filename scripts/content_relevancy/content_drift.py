from content_extraction.content_extraction import (
    get_relevant_content,
    get_embedded_content,
    is_page_unavailable,
)
from similarity.similarity import bert_similarity, combined_similarity
from summarization.summarization import summarize
from utils.utils import print_boxed_info
from bs4 import BeautifulSoup
import requests
from tqdm import tqdm
import pandas as pd
import time


def process_links(wikipedia_link, embedded_link):
    results = []
    wikipedia_content = get_relevant_content(
        wikipedia_link, "anchor_text"
    )  # Replace 'anchor_text' with actual anchor text if needed
    embedded_content = get_embedded_content(embedded_link)

    if is_page_unavailable(embedded_content):
        print(f"Page is unavailable: {embedded_link}")
        return None

    wikipedia_summary, embedded_summary = summarize(
        [wikipedia_content, embedded_content]
    )
    bert_relevancy = bert_similarity(wikipedia_summary, embedded_summary)
    combined_relevancy = combined_similarity(wikipedia_summary, embedded_summary)

    print_boxed_info(
        wikipedia_link,
        embedded_link,
        wikipedia_summary,
        embedded_summary,
        bert_relevancy,
        combined_relevancy,
    )

    results.append((wikipedia_link, combined_relevancy))
    return results


def get_wayback_links(embedded_link, num_versions=10, step_months=2):
    base_url = "http://web.archive.org/cdx/search/cdx"
    params = {
        "url": embedded_link,
        "output": "json",
        "fl": "timestamp,original",
        "limit": 1000,  # You can adjust this number
    }
    response = requests.get(base_url, params=params)
    items = response.json()[1:]
    items.reverse()

    selected_links = []
    prev_time = None

    for item in items:
        timestamp, original = item
        cur_time = time.strptime(timestamp[:6], "%Y%m")

        if (
            prev_time is None
            or (cur_time.tm_year - prev_time.tm_year) * 12
            + cur_time.tm_mon
            - prev_time.tm_mon
            >= step_months
        ):
            selected_links.append(original)
            prev_time = cur_time

            if len(selected_links) >= num_versions:
                break

    return selected_links


if __name__ == "__main__":
    wikipedia_link = "https://en.wikipedia.org/wiki/Strategic_National_Stockpile"
    embedded_link = "https://web.archive.org/web/20200403134414/https://www.phe.gov/about/sns/Pages/default.aspx"

    # Debug: Print the Wayback Machine links to make sure you have 10
    wayback_links = get_wayback_links(embedded_link)
    print(f"Total number of Wayback links: {len(wayback_links)}")
    print("Wayback links:", wayback_links)

    for wayback_link in wayback_links:
        # Debug: Print the Wikipedia content to make sure it's not empty
        print("Wikipedia content:", get_relevant_content(wikipedia_link, "anchor_text"))

        process_links(wikipedia_link, wayback_link)
