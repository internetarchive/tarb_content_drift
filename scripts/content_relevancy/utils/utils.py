def print_boxed_info(
    url,
    embedded_url,
    wikipedia_summary,
    embedded_summary,
    bert_similarity,
    combined_similarity,
):
    print("=" * 80)
    print("Wikipedia Link:\n" + url + "\n")
    print("Link:\n" + embedded_url + "\n")
    print("Wikipedia Content (being compared):\n" + wikipedia_summary[:400] + "...\n")
    print("Link Content:\n" + embedded_summary[:400] + "...\n")
    print(f"BERT Relevancy:\n{bert_similarity:.2f}%\n")
    print(f"Combined Model Relevancy:\n{combined_similarity:.2f}%\n")
    print("=" * 80 + "\n\n")
