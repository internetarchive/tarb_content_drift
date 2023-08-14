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
    weights = [0.5, 0.3, 0.2]
    return np.dot([bert_similarity, tfidf_similarity, jaccard_similarity], weights)
