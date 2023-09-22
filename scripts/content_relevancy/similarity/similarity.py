from transformers import BertModel, RobertaModel, XLNetModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
from tokenization.tokenization import tokenizer, roberta_tokenizer, xlnet_tokenizer
import numpy as np
import logging

# Set the logging level for the "transformers" logger to ERROR
logging.getLogger("transformers").setLevel(logging.ERROR)

# Suppress warnings
torch.set_grad_enabled(False)

# Define models
model = BertModel.from_pretrained("bert-base-uncased")
roberta_model = RobertaModel.from_pretrained("roberta-base")
xlnet_model = XLNetModel.from_pretrained("xlnet-base-cased")


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


def combined_similarity(text1, text2):
    # Get separate embeddings
    embeds1_bert = get_embeddings([text1], tokenizer, model)
    embeds2_bert = get_embeddings([text2], tokenizer, model)

    embeds1_roberta = get_embeddings([text1], roberta_tokenizer, roberta_model)
    embeds2_roberta = get_embeddings([text2], roberta_tokenizer, roberta_model)

    embeds1_xlnet = get_embeddings([text1], xlnet_tokenizer, xlnet_model)
    embeds2_xlnet = get_embeddings([text2], xlnet_tokenizer, xlnet_model)

    # Calculate cosine similarity separately
    bert_similarity = (
        torch.nn.functional.cosine_similarity(embeds1_bert, embeds2_bert).item() * 100
    )
    roberta_similarity = (
        torch.nn.functional.cosine_similarity(embeds1_roberta, embeds2_roberta).item()
        * 100
    )
    xlnet_similarity = (
        torch.nn.functional.cosine_similarity(embeds1_xlnet, embeds2_xlnet).item() * 100
    )

    weights = [0.5, 0.3, 0.2]  # Weights for BERT, RoBERTa, and XLNet respectively
    combined_similarity_score = np.dot(
        [bert_similarity, roberta_similarity, xlnet_similarity], weights
    )

    vectorizer = TfidfVectorizer(ngram_range=(1, 3))  # Using n-grams
    tfidf_matrix = vectorizer.fit_transform([text1, text2])

    tfidf_similarity = (
        cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0] * 100
    )

    jaccard_similarity = (
        len(set(text1.split()).intersection(set(text2.split())))
        / len(set(text1.split()).union(set(text2.split())))
    ) * 100

    # Adjust weights to focus more on exact text rather than semantic similarity
    weights = [0.5, 0.3, 0.2]

    return np.dot(
        [combined_similarity_score, tfidf_similarity, jaccard_similarity],
        weights,
    )
