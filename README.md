# tarb_gsoc23_content_drift

Content drift assessment tool for TARB project.
A Python script to analyze Wikipedia pages and calculate the relevancy scores of embedded non-wikipedia links.

## Directory Structure

- `scripts/`: Contains the main script `analysis.py`.
- `tests/`: Contains a file `test_links.txt` with a list of Wikipedia links to analyze.

## Usage

1. `pip install -r requirements.txt` to install the required libraries.
2. Add Wikipedia URLs to the `test_links.txt` file in the `tests` directory, one link per line.
3. Run the `wikipedia_analysis.py` script in the `scripts` directory:
