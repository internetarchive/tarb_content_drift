import openai
import streamlit as st

# Set up OpenAI API credentials
openai.api_key = "enter_api_key"

# Define the GPT-3.5 prompt
prompt = """
Given a URL and a context string, check the relevance of the URL to the context string.

URL: {url}
Description: {url_description}

Context: {context}
Description: {context_description}

Relevancy Percentage: """


# Function to generate descriptions using GPT-3.5
def generate_description(input_text):
    prompt = f"Describe the contents of the input:\n{input_text}\nDescription:"
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=50,
        n=1,
        stop=None,
        temperature=0.5,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    description = response.choices[0].text.strip()
    return description


# Function to check relevancy using GPT-3.5
def check_relevancy(url, context):
    # Generate response using GPT-3.5 to describe the URL
    url_description = generate_description(url)

    # Generate response using GPT-3.5 to describe the context
    context_description = generate_description(context)

    # Format the prompt with the URL and context
    formatted_prompt = prompt.format(
        url=url,
        url_description=url_description,
        context=context,
        context_description=context_description,
    )

    # Generate response using GPT-3.5
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=formatted_prompt,
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0.5,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )

    # Extract the relevancy percentage and description from the response
    relevancy_percentage = response.choices[0].text.strip()
    description = response.choices[0].text.strip()

    # Determine the reason for 0% relevancy
    reason = ""
    if relevancy_percentage == "0%":
        reason = "The URL and context are not related."

    return relevancy_percentage, description, reason


# Streamlit app
def main():
    # Get inputs from the user
    url = st.text_input("Enter the URL:")
    context = st.text_area("Enter the context:")

    # Check relevancy when the user clicks a button
    if st.button("Check Relevancy"):
        # Call the check_relevancy function
        relevancy_percentage, description, reason = check_relevancy(url, context)

        # Display the relevancy percentage, description, and reason
        st.markdown(
            f"""
        <div style="font-size:24px; color:lightblue">
            {description}
        </div>
        """,
            unsafe_allow_html=True,
        )


# Run the Streamlit app
if __name__ == "__main__":
    main()
