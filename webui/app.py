import time
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pyLDAvis.gensim
import numpy as np
import requests

st.set_page_config(layout="wide")

st.title("Relevancy Metrics")

webpage = st.text_input("Enter webpage URL", "")
context_string = st.text_area("Enter context string", "")

url = "http://localhost:5000/calculate_relevance"

if st.button("Calculate Relevance"):
    start_time = time.time()

    # Create a dictionary with your inputs to send as JSON
    payload = {"webpage": webpage, "context_string": context_string}
    try:
        response = requests.post(url, json=payload)
        # Check that request was successful
        response.raise_for_status()
    except requests.exceptions.HTTPError as errh:
        st.error(f"Http Error: {errh}")
        st.stop()
    except requests.exceptions.ConnectionError as errc:
        st.error(f"Error Connecting: {errc}")
        st.stop()
    except requests.exceptions.Timeout as errt:
        st.error(f"Timeout Error: {errt}")
        st.stop()
    except requests.exceptions.RequestException as err:
        st.error(f"Something went wrong: {err}")
        st.stop()

    # Parse the response JSON
    result = response.json()
    bert_similarity = float(result["bert_similarity"])
    gpt_similarity = result["gpt_similarity"]
    description = result["description"]

    end_time = time.time()
    runtime = end_time - start_time
    col1, col2, col3, col4 = st.columns(4)
    runtime = (end_time - start_time) * 1000

    with col1:
        st.subheader("BERT Similarity Score:")
        st.markdown(
            f"""
        <div style="font-size:24px; color:lightgreen">
            {bert_similarity * 100:.1f}%
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        st.subheader("GPT-3 Similarity score:")
        st.markdown(
            f"""
        <div style="font-size:24px; color:lightblue">
            {gpt_similarity}
        </div>
        """,
            unsafe_allow_html=True,
        )

    with st.sidebar:
        st.header("How does this work?")
        st.markdown(
            """
        This application uses the BERT (Bidirectional Encoder Representations from Transformers) model to measure the relevance between the text content of a provided webpage URL and a context string.
        """
        )

else:
    st.warning("Please enter both webpage URL and context string.")
