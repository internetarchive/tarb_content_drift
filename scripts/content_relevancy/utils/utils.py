def print_boxed_info(
    url,
    embedded_url,
    wikipedia_summary,
    embedded_summary,
    combined_similarity,
):
    # Define ANSI escape codes for colors
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    RESET = "\033[0m"

    print("=" * 80)
    print(GREEN + "Wikipedia Link:" + RESET)
    print(url + "\n")

    print(GREEN + "Link:" + RESET)
    print(embedded_url + "\n")

    print(YELLOW + "Wikipedia Content (being compared):" + RESET)
    print(wikipedia_summary[:400] + "...\n")

    print(YELLOW + "Link Content:" + RESET)
    print(embedded_summary[:400] + "...\n")

    print(BLUE + f"Semantic Relevance:" + RESET)
    print(f"{combined_similarity:.2f}%\n")

    print("=" * 80 + "\n\n")
