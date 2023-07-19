import numpy as np
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import gensim
from gensim import corpora, models, similarities
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.matutils import hellinger
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
import nltk
import os
import certifi
import openai

os.environ["SSL_CERT_FILE"] = certifi.where()

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "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"
openai.api_key = os.environ["OPENAI_API_KEY"]


def preprocess(text):
    result = []
    nltk_stopwords = set(stopwords.words("english"))
    for token in simple_preprocess(text):
        if token not in nltk_stopwords and len(token) > 3:
            result.append(WordNetLemmatizer().lemmatize(token, pos="v"))
    return result


def compute_lda_similarity(webpage, context_string):
    page = requests.get(webpage)
    soup = BeautifulSoup(page.content, "html.parser")

    webpage_text = "".join(node.get_text() for node in soup.find_all())
    webpage_tokens = preprocess(webpage_text)
    context_string_tokens = preprocess(context_string)

    dictionary = corpora.Dictionary([webpage_tokens, context_string_tokens])
    corpus = [
        dictionary.doc2bow(text) for text in [webpage_tokens, context_string_tokens]
    ]

    lda = models.LdaModel(
        corpus, num_topics=10, id2word=dictionary, passes=50, alpha="auto", eta="auto"
    )

    webpage_topic_distribution = lda.get_document_topics(
        corpus[0], minimum_probability=0
    )
    context_string_topic_distribution = lda.get_document_topics(
        corpus[1], minimum_probability=0
    )

    dense1 = gensim.matutils.sparse2full(webpage_topic_distribution, lda.num_topics)
    dense2 = gensim.matutils.sparse2full(
        context_string_topic_distribution, lda.num_topics
    )
    hellinger_distance = hellinger(dense1, dense2)

    return hellinger_distance


def compute_bert_relevance(webpage, context_string):
    page = requests.get(webpage)
    soup = BeautifulSoup(page.content, "html.parser")

    webpage_text = "".join(node.get_text() for node in soup.find_all())
    webpage_tokens = preprocess(webpage_text)
    context_string_tokens = preprocess(context_string)

    embedder = SentenceTransformer("bert-base-nli-mean-tokens")

    webpage_embedding = embedder.encode([" ".join(webpage_tokens)])[0]
    string_embedding = embedder.encode([" ".join(context_string_tokens)])[0]

    cosine_similarity = np.inner(webpage_embedding, string_embedding) / (
        np.linalg.norm(webpage_embedding) * np.linalg.norm(string_embedding)
    )
    hellinger_distance = compute_lda_similarity(webpage, context_string)
    return cosine_similarity, webpage_embedding, string_embedding, hellinger_distance


def compute_lda_model(webpage, context_string):
    page = requests.get(webpage)
    soup = BeautifulSoup(page.content, "html.parser")

    webpage_text = "".join(node.get_text() for node in soup.find_all())
    webpage_tokens = preprocess(webpage_text)
    context_string_tokens = preprocess(context_string)

    dictionary = corpora.Dictionary([webpage_tokens, context_string_tokens])
    corpus = [
        dictionary.doc2bow(text) for text in [webpage_tokens, context_string_tokens]
    ]

    lda = models.LdaModel(
        corpus, id2word=dictionary, num_topics=10, passes=50, alpha="auto", eta="auto"
    )

    coherence_model_lda = CoherenceModel(
        model=lda,
        texts=[webpage_tokens, context_string_tokens],
        dictionary=dictionary,
        coherence="c_v",
    )
    coherence_score = coherence_model_lda.get_coherence()

    return lda, corpus, dictionary, coherence_score


# Define the GPT-3.5 prompt
prompt = """
Given a URL and a context string, check the relevance of the URL to the context string.

URL: {url}
Description: {url_description}

Context: {context}
Description: {context_description}

Relevancy Percentage: """


# Function to generate descriptions using GPT-3.5
def generate_description(input_text):
    prompt = f"Describe the contents of the input:\n{input_text}\nDescription:"
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=50,
        n=1,
        stop=None,
        temperature=0.5,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    description = response.choices[0].text.strip()
    return description


# Function to check relevancy using GPT-3.5
def check_relevancy(url, context):
    # Generate response using GPT-3.5 to describe the URL
    url_description = generate_description(url)

    # Generate response using GPT-3.5 to describe the context
    context_description = generate_description(context)

    # Format the prompt with the URL and context
    formatted_prompt = prompt.format(
        url=url,
        url_description=url_description,
        context=context,
        context_description=context_description,
    )

    # Generate response using GPT-3.5
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=formatted_prompt,
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0.5,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )

    # Extract the relevancy percentage and description from the response
    relevancy_percentage = response.choices[0].text.strip()
    description = response.choices[0].text.strip()

    # Determine the reason for 0% relevancy
    reason = ""
    if relevancy_percentage == "0%":
        reason = "The URL and context are not related."

    return relevancy_percentage, description, reason
