from flask import Flask, request, jsonify
from content_extraction.content_extraction import (
    get_relevant_content,
    get_embedded_content,
)
from similarity.similarity import bert_similarity, combined_similarity
from summarization.summarization import summarize

# Initialize the Flask app
app = Flask(__name__)


# Modifying the relevancy endpoint to integrate existing functions
@app.route("/relevancy", methods=["POST"])
def get_relevancy():
    print("Received request")
    data = request.json
    link = data["link"]
    context = data["context"]

    print("Extracting Wikipedia content...")
    wikipedia_content = get_relevant_content(link, context)
    print("Extracting embedded content...")
    embedded_content = get_embedded_content(link)

    print("Summarizing content...")
    wikipedia_summary, embedded_summary = summarize(
        [wikipedia_content, embedded_content]
    )

    print("Calculating relevancy...")
    # Calculate bert_relevancy and combined_relevancy
    bert_relevancy = bert_similarity(wikipedia_summary, embedded_summary)
    combined_relevancy = combined_similarity(wikipedia_summary, embedded_summary)

    response = {
        "link": link,
        "context": context,
        "bert_relevancy": bert_relevancy,
        "combined_relevancy": combined_relevancy,
    }
    print("Response:", response)
    return jsonify(response)


# Run the app on port 5000
if __name__ == "__main__":
    print("Server is listening on http://127.0.0.1:5000")
    app.run(port=5000, debug=True)
