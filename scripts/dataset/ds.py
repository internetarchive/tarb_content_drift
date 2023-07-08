import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm
import concurrent.futures

# Do not process existing entries:
try:
    existing_df = pd.read_table(
        "new_file3.tsv",
        header=None,
        names=[
            "link",
            "wikipedia_page",
            "anchor_text",
            "page_heading",
            "surrounding_paragraph",
        ],
    )
except FileNotFoundError:
    existing_df = pd.DataFrame(
        columns=[
            "link",
            "wikipedia_page",
            "anchor_text",
            "page_heading",
            "surrounding_paragraph",
        ]
    )


# Function to process one row
def process_row(index, row):
    link, wikipedia_page = row["link"], row["wikipedia_page"]
    if not existing_df[
        (existing_df["link"] == link)
        & (existing_df["wikipedia_page"] == wikipedia_page)
    ].empty:
        # if it exists, we print a message and return None
        print(f"Skipping already processed entry: {link}, {wikipedia_page}")
        return None

    url = row["wikipedia_page"]
    if pd.isna(url):
        return index, row
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    # Find page heading
    page_heading = soup.find("h1")
    if page_heading:
        row["page_heading"] = page_heading.text

    # Find anchor text
    anchor = soup.find("a", href=row["link"])
    if anchor:
        row["anchor_text"] = anchor.text

        # Find surrounding paragraph
        paragraph = anchor.find_parent("p")
        if paragraph:
            row["surrounding_paragraph"] = paragraph.text

    return index, row


# Read the TSV file
df = pd.read_table("./test.tsv", header=None)

# Assign column names
df.columns = ["link", "wikipedia_page"]

# Create new columns
df["anchor_text"] = None
df["page_heading"] = None
df["surrounding_paragraph"] = None

# Use ThreadPoolExecutor to process rows in parallel
with concurrent.futures.ThreadPoolExecutor() as executor:
    for i in range(0, df.shape[0], 50):
        df_chunk = df.iloc[i : i + 50]
        # update the df_chunk with the returned data from executor.map
        result_list = list(
            filter(
                None,
                tqdm(
                    executor.map(
                        process_row, df_chunk.index, df_chunk.to_dict("records")
                    ),
                    total=df_chunk.shape[0],
                ),
            )
        )

        # Drop processed rows from the main DataFrame
        processed_indices = [result[0] for result in result_list]
        df.drop(processed_indices, inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Write the remaining DataFrame back to the original TSV file
        df.to_csv("./test.tsv", sep="\t", index=False, header=False)

        # Convert the remaining certain part of result_list, excluding the index
        df_chunk = pd.DataFrame(
            [result[1] for result in result_list], columns=df.columns
        )

        # Drop rows with any missing values
        df_chunk.dropna(inplace=True)

        # Check if file is empty before writing
        if i == 0:
            df_chunk.to_csv("output.tsv", sep="\t", index=False, mode="w", header=True)
        else:
            df_chunk.to_csv("output.tsv", sep="\t", index=False, mode="a", header=False)

# If there are any remaining rows, process them and write to the TSV file
if df.shape[0] % 50 != 0:
    df_chunk = df.iloc[(df.shape[0] // 50) * 50 :]
    result_list = list(
        tqdm(
            executor.map(process_row, df_chunk.index, df_chunk.to_dict("records")),
            total=df_chunk.shape[0],
        )
    )

    # Drop processed rows from the main DataFrame
    processed_indices = [result[0] for result in result_list]
    df.drop(processed_indices, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Convert the remaining certain part of result_list, excluding the index
    df_chunk = pd.DataFrame([result[1] for result in result_list], columns=df.columns)

    # Drop rows with any missing values
    df_chunk.dropna(inplace=True)

    # Check if file is empty before writing
    if df.shape[0] // 50 == 0:
        df_chunk.to_csv("output.tsv", sep="\t", index=False, mode="w", header=True)
    else:
        df_chunk.to_csv("output.tsv", sep="\t", index=False, mode="a", header=False)

print("Done")
