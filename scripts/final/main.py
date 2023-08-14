from content_extraction.content_extraction import (
    get_relevant_content,
    get_embedded_content,
    is_page_unavailable,
)
from similarity.similarity import bert_similarity, combined_similarity
from summarization.summarization import summarize
from utils.utils import print_boxed_info
from bs4 import BeautifulSoup
from tqdm import tqdm
import pandas as pd

df = pd.read_csv("./data/dataset_mini.tsv", sep="\t")
results = []

for i, row in tqdm(df.iterrows(), total=df.shape[0]):
    url = row["wikipedia_page"]
    anchor_text = row["anchor_text"]
    wikipedia_content = get_relevant_content(url, anchor_text)
    embedded_url = row["link"]
    embedded_content = get_embedded_content(embedded_url)

    if is_page_unavailable(embedded_content):
        print(f"Page is unavailable: {embedded_url}")
        continue

    wikipedia_summary, embedded_summary = summarize(
        [wikipedia_content, embedded_content]
    )
    bert_relevancy = bert_similarity(wikipedia_summary, embedded_summary)
    combined_relevancy = combined_similarity(wikipedia_summary, embedded_summary)

    print_boxed_info(
        url,
        embedded_url,
        wikipedia_summary,
        embedded_summary,
        bert_relevancy,
        combined_relevancy,
    )
    results.append((url, combined_relevancy))

with open("result.tsv", "w") as f:
    f.write("\t".join(["link", "relevancy_percentage"]) + "\n")
    for result in results:
        f.write("\t".join([str(r) for r in result]) + "\n")
