from flask import Flask, jsonify, request
from logic import compute_bert_relevance, check_relevancy, generate_description

app = Flask(__name__)


@app.route("/calculate_relevance", methods=["POST"])
def calculate_relevance():
    data = request.get_json()
    webpage = data.get("webpage")
    context_string = data.get("context_string")

    if webpage and context_string:
        try:
            # Calculate Bert and LDA relevance
            (
                bert_relevance,
                webpage_embedding,
                string_embedding,
                lda_relevance,
            ) = compute_bert_relevance(webpage, context_string)

            # Calculate GPT relevance and description
            (gpt_relevance, description, reason) = check_relevancy(
                webpage, context_string
            )

            result = {
                "bert_similarity": bert_relevance.tolist(),
                "lda_similarity": 1 - lda_relevance.tolist(),
                "gpt_similarity": gpt_relevance,
                "description": description,
                "reason": reason,
            }

            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e)})
    else:
        return jsonify({"error": "Please supply webpage and context_string"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
