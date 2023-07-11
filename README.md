# tarb_gsoc23_content_drift

Content drift assessment tool for TARB project.
A set of python scripts to analyze Wikipedia pages and calculate the relevancy scores of embedded non-wikipedia links.

## Directory Structure

- `scripts/`: Contains the scripts which have BERT and LDA models that calculate various metrics of content relevancy.
- `tests/`: Contains a file `test_links.txt` with a list of Wikipedia links to analyze.
- `data/`: Contains the TSV files with anchor texts, sub headings and surrounding paragraphs.
- `webui/`: Consists of a streamlit application that demonstrates how BERT calculates one of such relevancy metrics.

## Usage

1. `pip install -r requirements.txt` to install the required libraries.
2. Run the files in `scripts/` as required.
3. Instructions for running the webUI are in a separate readme within the folder.
