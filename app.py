import streamlit as st
import pandas as pd
import pdfplumber
import re
from fpdf import FPDF
import openai
import matplotlib.pyplot as plt
import nltk
from nltk.util import ngrams
from collections import Counter

# Ensure you have the NLTK data downloaded
nltk.download('punkt')

# Define weights for specific phrases
phrase_weights = {
    "net income": 1.5,
    "gross margin": 1.2,
    "operating expenses": 1.0,
    "free cash flow": 1.3,
    "earnings per share": 1.4,
    "capital expenditure": 1.1,
    "revenue growth": 1.2,
    "debt equity ratio": 1.0,
    "return on investment": 1.5,
    "profit margin": 1.3,
    "cost of goods sold": 1.1,
    "working capital": 1.2,
    "current ratio": 1.1,
    "quick ratio": 1.1,
    "interest coverage ratio": 1.2,
    "dividend yield": 1.3,
    "price to earnings ratio": 1.4,
    "asset turnover": 1.2,
    "inventory turnover": 1.1,
    "debt service coverage": 1.3,
    "return on equity": 1.5,
    "capital structure": 1.1,
    "liquidity ratio": 1.0,
    "cash flow from operations": 1.4,
    "net profit margin": 1.3,
    "total shareholder return": 1.2,
    "earnings before interest and taxes": 1.3
}



# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Load lexicon CSV file
lexicon_path = 'Loughran-McDonald_MasterDictionary_1993-2023.csv'
lexicon_df = pd.read_csv(lexicon_path)
lexicon_words = set(lexicon_df['Word'].str.lower())


# Function to clean text with selective n-grams and filtering
def clean_text_with_priority(text, lexicon_words, specific_phrases, phrase_weights, ngram_range=(1, 2)):
    # Convert to lowercase
    text = text.lower()

    # Remove unnecessary punctuation but keep financial symbols
    text = re.sub(r'[^\w\s$%]', '', text)

    # Tokenize text into words
    words = nltk.word_tokenize(text)

    # Generate unigrams and bigrams only
    all_ngrams = []
    for n in range(ngram_range[0], ngram_range[1] + 1):
        ngrams_list = list(ngrams(words, n))
        all_ngrams.extend(ngrams_list)
    
    # Convert n-grams to strings
    ngram_strings = [' '.join(ngram) for ngram in all_ngrams]

    # Filter and prioritize relevant n-grams, apply weights
    cleaned_ngrams = []
    weighted_phrases = []
    for ngram in ngram_strings:
        if ngram in lexicon_words or ngram in specific_phrases:
            if ngram in phrase_weights:
                # Apply the weight to the phrase by repeating it
                weighted_ngram = ' '.join([ngram] * int(phrase_weights[ngram] * 10))
                weighted_phrases.append(weighted_ngram)
            cleaned_ngrams.append(ngram)
        elif ngram in words:  # Preserve unigrams
            cleaned_ngrams.append(ngram)

    # Join the cleaned n-grams back into a cleaned text
    cleaned_text = ' '.join(cleaned_ngrams + weighted_phrases)

    return cleaned_text


# Function to get sentiment analysis using OpenAI
def get_sentiment_analysis(text, temperature=0.1):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an expert Financial Analyst conducting trading sentiment analysis."},
            {"role": "user", "content": f"Analyze the text and financial data for immediate trading sentiment. Evaluate and outweigh each point whose significance for short-term and immediate impacts is high. Determine the accurate immediate trading sentiment (positive, negative, neutral) based on weighted assessment. Justify the conclusion in 2 sentences, show conclusion first.\n\n\n\n\n\n{text}"}
        ],
        temperature=temperature
    )
    return response['choices'][0]['message']['content'].strip()

# Function to visualize sentiment analysis results
def visualize_sentiments(sentiments):
    # Count the occurrence of each sentiment
    sentiment_counts = pd.Series(sentiments).value_counts()
    
    # Plot the sentiment counts as a bar chart
    fig, ax = plt.subplots()
    sentiment_counts.plot(kind='bar', ax=ax)
    ax.set_title('Sentiment Analysis Results')
    ax.set_xlabel('Sentiment')
    ax.set_ylabel('Count')
    st.pyplot(fig)

# Streamlit app with sidebar for API key input
def main():
    st.title("PDF Sentiment Analyzer ChatBot 📖")

    # Sidebar for API key input
    with st.sidebar:
        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
        st.markdown("[Get an OpenAI API key](https://platform.openai.com/account/api-keys)")

    # Check if the API key has been provided
    if not openai_api_key:
        st.error("Please enter your OpenAI API key in the sidebar.")
        return

    # Set up OpenAI API key
    openai.api_key = openai_api_key
    
    # Select phrases to prioritize
    selected_phrases = st.multiselect(
        "Select specific financial phrases to prioritize:",
        options=list(phrase_weights.keys()),
        default=list(phrase_weights.keys())
    )

    # Update the specific_phrases set based on user selection
    specific_phrases = set(selected_phrases)

    # Adjust the temperature parameter for the sentiment analysis
    temperature = st.slider("Select temperature for sentiment analysis (lower is more deterministic):", 0.0, 1.0, 0.1)

    
    # Text input
    user_input = st.text_area("Enter your text here:")

    # PDF file upload
    uploaded_files = st.file_uploader("Or upload PDF files", type="pdf", accept_multiple_files=True)

    all_sentiments = []

    if user_input:
        st.write("Processing entered text...")
        # Clean the entered text
        with st.spinner("Cleaning text..."):
            cleaned_text = clean_text_with_priority(user_input, lexicon_words, specific_phrases, phrase_weights)

        # Perform sentiment analysis on the cleaned text
        with st.spinner("Analyzing sentiment..."):
            sentiment_analysis = get_sentiment_analysis(cleaned_text)
            all_sentiments.append(sentiment_analysis)

        # Display sentiment analysis result
        st.subheader("Sentiment Analysis")
        st.write(sentiment_analysis)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            st.write(f"Processing {uploaded_file.name}...")

            # Extract text from the uploaded PDF
            with st.spinner("Extracting text from PDF..."):
                extracted_text = extract_text_from_pdf(uploaded_file)

            # Clean the extracted text
            with st.spinner("Cleaning text..."):
                cleaned_text = clean_text_with_priority(user_input, lexicon_words, specific_phrases, phrase_weights)

            # Perform sentiment analysis on the cleaned text
            with st.spinner("Analyzing sentiment..."):
                sentiment_analysis = get_sentiment_analysis(cleaned_text)
                all_sentiments.append(sentiment_analysis)

            # Display the sentiment analysis result
            st.subheader(f"Sentiment Analysis for {uploaded_file.name}")
            st.write(sentiment_analysis)

            # Create and download cleaned PDF
            class PDF(FPDF):
                def body(self, body):
                    self.set_font('Arial', '', 12)
                    self.multi_cell(0, 10, body)
                    self.ln()

            cleaned_pdf = PDF()
            cleaned_pdf.add_page()
            cleaned_pdf.body(cleaned_text)
            output_pdf_path = f'cleaned_output_{uploaded_file.name}'
            cleaned_pdf.output(output_pdf_path)

            with open(output_pdf_path, "rb") as file:
                st.download_button(label=f"Download Cleaned PDF for {uploaded_file.name}", data=file, file_name=output_pdf_path)

    # Visualize the overall sentiment analysis results if any
    if all_sentiments:
        st.subheader("Overall Sentiment Analysis")
        visualize_sentiments(all_sentiments)


if __name__ == "__main__":
    main()
