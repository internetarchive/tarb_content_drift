import time
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from logic import compute_bert_relevance

st.set_page_config(layout="wide")

st.title("BERT Relevance Calculator")

webpage = st.text_input("Enter webpage URL", "")
context_string = st.text_area("Enter context string", "")

if st.button("Calculate Relevance"):
    if webpage and context_string:
        start_time = time.time()
        relevance, webpage_embedding, string_embedding = compute_bert_relevance(
            webpage, context_string
        )
        end_time = time.time()
        runtime = end_time - start_time
        col1, col2, col3 = st.columns(3)
        runtime = (end_time - start_time) * 1000

        with col1:
            st.subheader("Relevance Score:")
            st.markdown(
                f"""
            <div style="font-size:24px; color:lightgreen">
                {relevance * 100:.1f}%
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col2:
            st.subheader("Runtime:")
            st.markdown(
                f"""
            <div style="font-size:24px; color:pink">
                {runtime:.2f} ms
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

            embed_col1, embed_col2 = st.columns(2)
            with embed_col1:
                st.markdown(
                    "<h4 style='font-size:12px;'>Webpage Embedding:</h4>",
                    unsafe_allow_html=True,
                )
                st.write(webpage_embedding)
            with embed_col2:
                st.markdown(
                    "<h4 style='font-size:12px;'>Context String Embedding:</h4>",
                    unsafe_allow_html=True,
                )
                st.write(string_embedding)

            embeddings = np.stack([webpage_embedding, string_embedding])
            pca = PCA(n_components=2)
            embeddings_2d = pca.fit_transform(embeddings)

            st.write("Here's a plot to visualize how the embeddings are compared:")
            fig, ax = plt.subplots()
            ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], color="blue")
            ax.plot(embeddings_2d[:, 0], embeddings_2d[:, 1], color="red")

            for i, txt in enumerate(["Webpage Embedding", "Context String Embedding"]):
                ax.annotate(
                    txt,
                    (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                    textcoords="offset points",
                    xytext=(10, 10),
                    ha="center",
                )

            ax.axhline(0, color="black", linewidth=0.5)
            ax.axvline(0, color="black", linewidth=0.5)
            ax.grid(color="gray", linestyle="--", linewidth=0.5)
            st.pyplot(fig)

    else:
        st.warning("Please enter both webpage URL and context string.")
