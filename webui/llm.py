import openai
import requests
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
import streamlit as st

# Set up OpenAI API credentials
openai.api_key = "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"


def count_tokens(text):
    """
    Counts the number of tokens in the given text.

    Args:
    text (str): Text to count tokens.

    Returns:
    count (int): Number of tokens.
    """
    tokens = word_tokenize(text)
    count = len(tokens)
    return count


def extract_url_content(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    for script in soup(["script", "style"]):
        script.decompose()
    text = soup.get_text()
    content = " ".join(text.split())
    tokens = word_tokenize(content)
    if len(tokens) > 2500:  # Added condition to handle large content
        content = summarize_text_batches(content)
    return content


def summarize_text_batches(text):
    """
    Summarizes the given text in batches.

    Args:
    text (str): Text to be summarized.

    Returns:
    summary (str): Summarized text.
    """
    tokens = word_tokenize(text)
    # Define the maximum number of tokens in a chunk
    max_tokens = 4096
    # Split the tokens into chunks
    chunks = [tokens[i : i + max_tokens] for i in range(0, len(tokens), max_tokens)]
    # Summarize each chunk and combine the summaries
    summaries = [summarize_chunk(chunk) for chunk in chunks]
    summary = " ".join(summaries)
    return summary


def summarize_chunk(chunk):
    """
    Summarizes a chunk of text using GPT-3.

    Args:
    chunk (list): A chunk of tokens.

    Returns:
    summary (str): Summarized chunk of text.
    """
    prompt = f"Describe the contents of the input:\n{' '.join(chunk)}"
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
    summary = response.choices[0].text.strip()
    return summary


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
    # Extract URL content
    url_content = extract_url_content(url)

    # Check if content exceeds 4096 tokens
    if count_tokens(url_content) > 4096:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=f"Summarize the following content:\n{url_content}",
            temperature=0.3,
            max_tokens=200,
        )
        url_content = response.choices[0].text.strip()

    # Describe the URL content and context
    url_description = generate_description(url_content)
    context_description = generate_description(context)

    # Format the prompt with the URL and context
    formatted_prompt = f"URL: {url}\nURL Description: {url_description}\nContext: {context}\nContext Description: {context_description}\nRelevancy Check:"

    # Generate response using GPT-3.5
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=formatted_prompt,
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0.5,
    )

    # Extract the relevancy percentage from the response
    relevancy_percentage = response.choices[0].text.strip()

    return {
        "relevancy_percentage": relevancy_percentage,
        "url_description": url_description,
        "context_description": context_description,
    }


# Streamlit app
def main():
    # Get inputs from the user
    url = st.text_input("Enter the URL:")
    context = st.text_area("Enter the context:")

    # Check relevancy when the user clicks a button
    if st.button("Check Relevancy"):
        # Call the check_relevancy function
        result = check_relevancy(url, context)

        # Display the relevancy percentage, description, and reason
        st.markdown(f"**Relevancy Percentage:** {result['relevancy_percentage']}")
        st.markdown(f"**Description of URL:** {result['url_description']}")
        st.markdown(f"**Description of Context:** {result['context_description']}")
        st.markdown(f"**Reason for Relevancy Percentage:** {result['reason']}")


# Run the Streamlit app
if __name__ == "__main__":
    main()
