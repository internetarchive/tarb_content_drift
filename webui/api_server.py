from flask import Flask, jsonify, request
from logic import compute_bert_relevance

app = Flask(__name__)


@app.route("/calculate_relevance", methods=["POST"])
def calculate_relevance():
    data = request.get_json()
    webpage = data.get("webpage")
    context_string = data.get("context_string")
    if webpage and context_string:
        try:
            relevance, webpage_embedding, string_embedding = compute_bert_relevance(
                webpage, context_string
            )
            result = {
                "relevance": relevance.tolist(),
            }
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e)}), 400
    else:
        return jsonify({"error": "Please supply webpage and context_string"}), 400


if __name__ == "__main__":
    app.run(debug=True, port=5000)
