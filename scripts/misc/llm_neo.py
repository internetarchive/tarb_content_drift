import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from transformers import (
    BartTokenizer,
    BartForConditionalGeneration,
    BertTokenizer,
    BertModel,
    RobertaTokenizer,
    RobertaModel,
    XLNetTokenizer,
    XLNetModel,
)
import torch
import numpy as np
from readability import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

# Set the logging level for the "transformers" logger to ERROR
logging.getLogger("transformers").setLevel(logging.ERROR)

# Suppress warnings
torch.set_grad_enabled(False)

# Model and tokenizer for summarization
summarizer_model = BartForConditionalGeneration.from_pretrained(
    "facebook/bart-large-cnn"
)
summarizer_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

# BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

# Additional Models
roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
roberta_model = RobertaModel.from_pretrained("roberta-base")
xlnet_tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
xlnet_model = XLNetModel.from_pretrained("xlnet-base-cased")

# HTTP Session
session = requests.Session()


def get_combined_embeddings(texts):
    bert_embeds = get_embeddings(texts, tokenizer, model)
    roberta_embeds = get_embeddings(texts, roberta_tokenizer, roberta_model)
    xlnet_embeds = get_embeddings(texts, xlnet_tokenizer, xlnet_model)
    return torch.cat((bert_embeds, roberta_embeds, xlnet_embeds), dim=1)


def get_embeddings(texts, tokenizer, model):
    inputs = tokenizer(
        texts, return_tensors="pt", max_length=512, truncation=True, padding=True
    )
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)


def bert_similarity(text1, text2):
    embeds1 = get_embeddings([text1], tokenizer, model)
    embeds2 = get_embeddings([text2], tokenizer, model)
    return torch.nn.functional.cosine_similarity(embeds1, embeds2).item() * 100


def combined_similarity(text1, text2):
    embeds1 = get_combined_embeddings([text1])
    embeds2 = get_combined_embeddings([text2])
    bert_similarity = (
        torch.nn.functional.cosine_similarity(embeds1, embeds2).item() * 100
    )

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    tfidf_similarity = (
        cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0] * 100
    )

    jaccard_similarity = (
        len(set(text1.split()).intersection(set(text2.split())))
        / len(set(text1.split()).union(set(text2.split())))
    ) * 100

    # BERT Similarity: BERT captures deep semantic relationships between texts, which can lead to higher similarity scores when the texts are semantically similar even if the exact words differ.
    # TF-IDF Similarity: This measure relies on term frequency and the inverse document frequency of words. If the texts have different words but the same meaning, the TF-IDF similarity may be low, reducing the overall combined similarity.
    # Jaccard Similarity: Jaccard similarity depends on the intersection of words between the two texts. If the words are different but the meaning is the same, the Jaccard similarity may be low.

    # Adjust weights
    weights = [0.4, 0.3, 0.3]
    return np.dot([bert_similarity, tfidf_similarity, jaccard_similarity], weights)


def is_page_unavailable(content):
    unavailable_phrases = [
        "This Page is Unavailable Right Now",
        "404 Not Found",
        "Page Not Found",
    ]
    return any(phrase in content for phrase in unavailable_phrases)


def summarize(texts):
    summaries = []
    for text in texts:
        chunks = [text[i : i + 1024] for i in range(0, len(text), 1024)]
        summarized_chunks = ""
        for chunk in chunks:
            inputs = summarizer_tokenizer(
                chunk, return_tensors="pt", max_length=1024, truncation=True
            )
            summary_ids = summarizer_model.generate(
                inputs.input_ids,
                max_length=150,
                min_length=50,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True,
            )
            summarized_chunks += (
                summarizer_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                + " "
            )
        summaries.append(summarized_chunks.strip())
    return summaries


def get_relevant_content(url, anchor_text):
    response = session.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    anchor_tag = soup.find("a", string=anchor_text)
    if anchor_tag:
        relevant_container = anchor_tag.find_parent(["p", "div", "li"])
        return relevant_container.get_text() if relevant_container else ""
    return ""


def get_embedded_content(url):
    response = session.get(url)
    doc = Document(response.text)
    summary_html = doc.summary()
    soup = BeautifulSoup(summary_html, "html.parser")
    return soup.get_text()


def print_boxed_info(
    url,
    embedded_url,
    wikipedia_summary,
    embedded_summary,
    bert_similarity,
    combined_similarity,
):
    print("=" * 80)
    print("Wikipedia Link:\n" + url + "\n")
    print("Link:\n" + embedded_url + "\n")
    print("Wikipedia Content (being compared):\n" + wikipedia_summary[:400] + "...\n")
    print("Link Content:\n" + embedded_summary[:400] + "...\n")
    print(f"BERT Relevancy:\n{bert_similarity:.2f}%\n")
    print(f"Combined Model Relevancy:\n{combined_similarity:.2f}%\n")
    print("=" * 80 + "\n\n")


def main():
    df = pd.read_csv("../data/dataset_mini.tsv", sep="\t")
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


if __name__ == "__main__":
    main()
