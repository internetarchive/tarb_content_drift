from flask import Flask, request, jsonify
from content_extraction.content_extraction import (
    get_relevant_content,
    get_embedded_content,
)
from similarity.similarity import bert_similarity, combined_similarity
from summarization.summarization import summarize

# Initialize the Flask app
app = Flask(__name__)


# Modifying the get_relevancy endpoint to integrate existing functions
@app.route("/get_relevancy", methods=["POST"])
def get_relevancy():
    data = request.json
    link = data["link"]
    context = data["context"]

    wikipedia_content = get_relevant_content(link, context)
    embedded_content = get_embedded_content(link)

    wikipedia_summary, embedded_summary = summarize(
        [wikipedia_content, embedded_content]
    )

    # Calculate bert_relevancy and combined_relevancy
    bert_relevancy = bert_similarity(wikipedia_summary, embedded_summary)
    combined_relevancy = combined_similarity(wikipedia_summary, embedded_summary)

    return jsonify(
        {
            "link": link,
            "context": context,
            "bert_relevancy": bert_relevancy,
            "combined_relevancy": combined_relevancy,
        }
    )


# Run the app on port 5000
if __name__ == "__main__":
    app.run(port=5000, debug=True)
