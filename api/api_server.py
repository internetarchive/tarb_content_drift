from bert_api import compute_bert_relevance
from lda_api import find_closest_content_lda
from llm_api import check_relevancy
from flask import Flask, jsonify, request

app = Flask(__name__)


@app.route("/relevance", methods=["POST"])
def calculate_relevance():
    data = request.get_json()
    webpage = data.get("webpage")
    context_string = data.get("context_string")

    if webpage and context_string:
        try:
            # BERT Relevance
            bert_relevance = compute_bert_relevance(webpage, context_string)

            # LDA Relevance
            lda_relevance = find_closest_content_lda(webpage, context_string)

            # LLM Relevance
            # llm_relevance = check_relevancy(webpage, context_string)

            return jsonify(
                {
                    "bert_relevance": float(bert_relevance),
                    "lda_relevance": float(lda_relevance),
                    # "llm_relevance": llm_relevance,
                }
            )
        except Exception as e:
            return jsonify({"error": str(e)}), 400
    else:
        return jsonify({"error": "Please supply webpage and context_string"}), 400


if __name__ == "__main__":
    app.run(port=5000)
