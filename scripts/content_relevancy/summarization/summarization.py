from transformers import BartTokenizer, BartForConditionalGeneration

# Define summarizer models
summarizer_model = BartForConditionalGeneration.from_pretrained(
    "facebook/bart-large-cnn"
)
summarizer_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")


# Summarize text
def summarize(texts):
    summaries = []
    for text in texts:
        chunks = [text[i : i + 1024] for i in range(0, len(text), 1024)]
        summarized_chunks = ""
        for chunk in chunks:
            inputs = summarizer_tokenizer(
                chunk, return_tensors="pt", max_length=1024, truncation=True
            )
            summary_ids = summarizer_model.generate(
                inputs.input_ids,
                max_length=150,
                min_length=50,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True,
            )
            summarized_chunks += (
                summarizer_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                + " "
            )
        summaries.append(summarized_chunks.strip())
    return summaries
