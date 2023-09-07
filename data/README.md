# DATA

## Overview

The `data` directory contains essential datasets used for analyzing Wikipedia pages and calculating the relevancy scores of embedded non-Wikipedia links. The datasets are stored in TSV (Tab-Separated Values) format.

## Files

### `rawdata.tsv`

This file contains the raw dataset created from Wikipedia's EventStream data. It serves as the initial data source for further processing and analysis.

```plaintext
File: `rawdata.tsv`

File: `dataset.tsv`
Columns: `link`, `wikipedia_page`, `anchor_text`, `page_heading`, `surrounding_paragraph`
