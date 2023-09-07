# tarb_gsoc23_content_drift

## Overview

This repository houses the Content Drift Assessment Tool developed for the TARB project at the Internet Archive. The tool comprises a collection of Python scripts designed to analyze Wikipedia pages and compute relevancy scores for embedded non-Wikipedia links using advanced NLP models like BERT and LDA.

## Directory Structure

- `scripts/`: This directory contains Python scripts that utilize BERT and LDA models to calculate various metrics related to content relevancy. These metrics are crucial for understanding how well the embedded links align with the content of the Wikipedia pages.
  
- `api/`: This directory hosts APIs that expose the various relevancy metrics calculated by the models. These APIs can be integrated into other systems or used for batch processing.
  
- `data/`: This directory contains TSV (Tab Separated Values) files that store anchor texts, sub-headings, and surrounding paragraphs for each analyzed Wikipedia page. These files serve as the data foundation for the relevancy calculations.
  
- `webui/`: This directory features a Streamlit application that provides a user-friendly interface to interact with the BERT model for calculating relevancy metrics. It serves as a demo to showcase the capabilities of the tool.

## Prerequisites

- Python 3.x
- pip

## Installation and Usage

1. Clone the repository to your local machine.
