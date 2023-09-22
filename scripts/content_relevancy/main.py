from content_extraction.content_extraction import (
    get_relevant_content,
    get_embedded_content,
    is_page_unavailable,
)
from similarity.similarity import (
    combined_similarity,
)
from summarization.summarization import summarize
from utils.utils import print_boxed_info
from bs4 import BeautifulSoup
from tqdm import tqdm
import pandas as pd

# Load dataset
print("Loading dataset...")
df = pd.read_csv("./data/dataset_mini.tsv", sep="\t")
results = []

print("Starting iteration over DataFrame rows...")
for i, row in tqdm(df.iterrows(), total=df.shape[0]):
    print(f"Processing row {i + 1}")

    url = row["wikipedia_page"]
    anchor_text = row["anchor_text"]
    surrounding_paragraph = row["surrounding_paragraph"]

    link_url = row["link"]
    link_content = get_embedded_content(link_url)

    print("Checking if linked page is available...")
    if is_page_unavailable(link_content):
        print(f"Page is unavailable: {link_url}")
        continue

    # Summarize the contents
    # print("Summarizing Wikipedia and linked page content...")
    # wikipedia_summary, link_summary = summarize([wikipedia_content, link_content])

    # Compute relevancy scores
    print("Computing relevancy scores...")

    combined_relevancy = combined_similarity(wikipedia_content, link_content)

    print_boxed_info(
        url,
        link_url,
        wikipedia_content,
        link_content,
        combined_relevancy,
    )

    # Append results
    print("Appending results to the list...")
    results.append((url, combined_relevancy))

# Write results to a TSV file
print("Writing results to a TSV file...")
with open("result.tsv", "w") as f:
    f.write("\t".join(["link", "relevancy_percentage"]) + "\n")
    for result in results:
        f.write("\t".join([str(r) for r in result]) + "\n")

print("Process completed.")
