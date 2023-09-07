# API

## Overview

This API is part of the TARB project developed for Google Summer of Code 2023. It provides endpoints to calculate the relevancy of content on Wikipedia pages using different models like BERT, LDA, and LLM.

## API Endpoints

The API exposes three main endpoints for calculating content relevancy:

1. **BERT Relevance**: Calculates the relevancy score using the BERT model.
   `curl -XPOST -H "Content-type: application/json" -d '{"webpage": "https://example.com", "context_string": "your context string"}' 'localhost:5000/bert_relevance'`
3. **LDA Relevance**: Utilizes the LDA model for relevancy calculation.
   `curl -XPOST -H "Content-type: application/json" -d '{"webpage": "https://example.com", "context_string": "your context string"}' 'localhost:5001/lda_relevance'`
5. **LLM Relevance**: Uses the LLM model to determine content relevancy.
   `curl -XPOST -H "Content-type: application/json" -d '{"webpage": "https://example.com", "context_string": "your context string"}' 'localhost:5002/llm_relevance'`

   ![CleanShot 2023-07-11 at 09 09 34@2x](https://github.com/internetarchive/tarb_gsoc23_content_drift/assets/63366288/4986a970-1096-498e-b542-30d03baa0138)

## Prerequisites

- Docker (optional)
- Python 3.x
- pip

## Setup and Running the API

### Using Docker (Optional)

If you have Docker installed, you can build and run the API as a Docker container.

```bash
docker build -t tarb_api .
docker run -p 5000-5002:5000-5002 tarb_api
