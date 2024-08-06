import streamlit as st
import pdfplumber
import pandas as pd
import re
from fpdf import FPDF
from dotenv import load_dotenv
import os
import openai
import matplotlib.pyplot as plt

# Load environment variables from a .env file if present
load_dotenv()

# Set up OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Function to clean text with priority given to lexicon words
def clean_text_with_priority(text, lexicon_words):
    # Convert to lowercase
    text = text.lower()

    # Remove unnecessary punctuation but keep financial symbols
    text = re.sub(r'[^\w\s$%]', '', text)

    # Tokenize text
    words = text.split()

    # Prioritize and preserve lexicon words
    cleaned_words = []
    for word in words:
        if word in lexicon_words:
            cleaned_words.append(word)
        else:
            # Optionally, you can apply further cleaning to non-lexicon words
            cleaned_words.append(word)

    # Join the words back into a cleaned text
    cleaned_text = ' '.join(cleaned_words)

    return cleaned_text

# Function to get sentiment analysis using OpenAI
def get_sentiment_analysis(text, temperature=0.1):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an expert Financial Analyst conducting trading sentiment analysis."},
            {"role": "user", "content": f"Analyze the text for immediate trading sentiment. Evaluate and outweigh each point whose significance for short-term and immediate impacts is high. Determine the accurate immediate trading sentiment (positive, negative, neutral) based on weighted assessment. Justify the conclusion in 2 sentences.\n\n\n\n\n\n{text}"}
        ],
        temperature=temperature
    )
    return response['choices'][0]['message']['content'].strip()

# Load lexicon CSV file
lexicon_path = '/Users/nick/Downloads/Loughran-McDonald_MasterDictionary_1993-2023.csv'
lexicon_df = pd.read_csv(lexicon_path)
lexicon_words = set(lexicon_df['Word'].str.lower())

# Function to visualize sentiment analysis results
def visualize_sentiments(sentiments):
    sentiment_counts = pd.Series(sentiments).value_counts()
    fig, ax = plt.subplots()
    sentiment_counts.plot(kind='bar', ax=ax)
    ax.set_title('Sentiment Analysis Results')
    ax.set_xlabel('Sentiment')
    ax.set_ylabel('Count')
    st.pyplot(fig)

# Streamlit app
def main():
    st.title("PDF Sentiment Analyzer ChatBot 📖")

    # Text input
    user_input = st.text_area("Enter your text here:")

    # PDF file upload
    uploaded_files = st.file_uploader("Or upload PDF files", type="pdf", accept_multiple_files=True)

    all_sentiments = []

    if user_input:
        st.write("Processing entered text...")
        # Clean the entered text
        with st.spinner("Cleaning text..."):
            cleaned_text = clean_text_with_priority(user_input, lexicon_words)

        # Perform sentiment analysis on the cleaned text
        with st.spinner("Analyzing sentiment..."):
            sentiment_analysis = get_sentiment_analysis(cleaned_text)
            all_sentiments.append(sentiment_analysis)

        # Display the cleaned text and sentiment analysis result
        st.subheader("Cleaned Text")
        st.write(cleaned_text)

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
                cleaned_text = clean_text_with_priority(extracted_text, lexicon_words)

            # Display the cleaned text
            st.subheader(f"Cleaned Text from {uploaded_file.name}")
            st.write(cleaned_text)

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
