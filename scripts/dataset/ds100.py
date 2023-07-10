from concurrent.futures import ThreadPoolExecutor
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

CHUNK_SIZE = 100  # increase chunk size; experiment with this value to find a sweet spot
WORKERS = 50  # we'll use 50 workers to exploit I/O-bound concurrency; adjust this according to your specific situation

session = requests.Session()  # use a session to re-use the same connection


def process_row(row):
    try:
        response = session.get(row["wikipedia_page"])  # use the session
        soup = BeautifulSoup(response.text, "html.parser")

        page_heading = soup.find("h1")
        if page_heading:
            row["page_heading"] = page_heading.text

        anchor = soup.find("a", href=row["link"])
        if anchor:
            row["anchor_text"] = anchor.text
            # find surrounding paragraph
            paragraph = anchor.find_parent("p")
            if paragraph:
                row["surrounding_paragraph"] = paragraph.text

        return row
    except Exception as e:
        print(f"Error processing {row['wikipedia_page']}: {e}")
        return np.nan


df = pd.read_table("./test.tsv", header=None)

# Assign column names
df.columns = ["link", "wikipedia_page"]

# Create new columns
df["anchor_text"] = np.nan
df["page_heading"] = np.nan
df["surrounding_paragraph"] = np.nan

# Process data by chunks using ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=WORKERS) as executor:
    for i in range(0, df.shape[0], CHUNK_SIZE):
        print(f"Processing batch {i//CHUNK_SIZE + 1}...")
        df_chunk = df.iloc[i : i + CHUNK_SIZE]

        results = list(
            tqdm(
                executor.map(process_row, df_chunk.to_dict("records")),
                total=df_chunk.shape[0],
            )
        )

        # Store the processed chunk in a DataFrame
        df_processed = pd.DataFrame(results)

        # Drop rows with missing values
        df_processed.dropna(inplace=True)

        # Write processed data to TSV
        if df_processed.shape[0] > 0:
            df_processed.to_csv(
                "output.tsv", sep="\t", index=False, mode="a", header=not bool(i)
            )

print("Finished processing.")



