import streamlit as st
import pandas as pd
import pdfplumber
import re
from fpdf import FPDF
import openai
import matplotlib.pyplot as plt
import nltk
import base64

from nltk.util import ngrams
from collections import Counter

# Set the NLTK data path to your 'nltk_data' directory
nltk_data_path = "nltk_data"

# Append this path to NLTK's data search paths
nltk.data.path.append('./nltk_data')

nltk.download('punkt_tab')

from nltk.tokenize import word_tokenize

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

    # Generate unigrams and bigrams 
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

# Function to get sentiment analysis using OpenAI for PDFs and text input
def get_sentiment_analysis(text, temperature=0.05):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an expert Financial Analyst conducting trading sentiment analysis with a focus on negative financial impacts."},
            {"role": "user", "content": f"Analyze the text for immediate trading sentiment. Mention positive,negative and neutral statements with necessary details. Evaluate and outweigh each point whose significance for short-term and immediate impacts is high. Determine the accurate immediate trading sentiment (positive, negative, neutral) based on weighted assessment. Justify the conclusion. Show the conclusion first. Display the conclusion first for each file. Keep the output format consistent for every file.\n\n\n\n\n\n{text}"}
        ],
        temperature=temperature
    )
    sentiment_text = response['choices'][0]['message']['content'].strip()
    
    # Extract a continuous sentiment score from the sentiment analysis
    score = sentiment_text.count('positive') * 1 - sentiment_text.count('negative') * 1
    return sentiment_text, score

# Function to plot sentiment scores line chart
def plot_sentiment_scores(df):
    fig, ax = plt.subplots()
    ax.plot(df['PDF Name'], df['Sentiment Score'], marker='o')
    ax.set_xlabel('PDF Name')
    ax.set_ylabel('Sentiment Score')
    ax.set_title('Sentiment Scores for PDFs')  # Corrected method
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)


# Streamlit app configuration
st.set_page_config(
    page_title="Financial Sentiment Analyzer",
    page_icon="✔",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "# This is a Financial Sentiment Analyzer app created with Streamlit"
    }
)

# Function to encode an image to base64
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Encode your background images
img_main = get_img_as_base64("Background3.png")

# Custom CSS for background images and styling
def add_custom_css():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{img_main}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        .css-1d391kg {{
            background-color: rgba(255, 255, 255, 0.85);
            border-radius: 10px;
            padding: 20px;
        }}
        .stButton > button {{
            background-color: #1f77b4;
            color: white;
            border-radius: 10px;
            height: 50px;
            width: 100%;
            font-size: 18px;
            border: none;
        }}
        .stButton > button:hover {{
            background-color: #0056a1;
        }}
        .stTextInput > div > div > input {{
            font-size: 18px;
            padding: 10px;
            background-color: #ffffff;
            border-radius: 10px;
            border: 2px solid #000000;
        }}
        .stFileUploader > div > div > div > button {{
            font-size: 18px;
            border-radius: 10px;
            background-color: #1f77b4;
            color: white;
            border: none;
        }}
        .stFileUploader > div > div > div > button:hover {{
            background-color: #0056a1;
        }}
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {{
            color: #000000;
        }}
        .css-1aumxhk, .css-1avcm0n, .css-1kyxreq, .css-1d391kg, .css-1offfwp, .css-pkbazv {{
            background-color: rgba(255, 255, 255, 0.85) !important;
            border-radius: 10px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_custom_css()

# Define the PDF class here so it's available throughout the script
class PDF(FPDF):
    def body(self, body):
        self.set_font('Arial', '', 14)
        # Handling utf-8 encoded strings
        body = body.encode('latin-1', 'replace').decode('latin-1')
        self.multi_cell(0, 10, body)
        self.ln()

# Streamlit app main function
def main():
    st.title("Financial Sentiment Analyzer ✔")

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

    st.markdown("## Analyze PDFs and Text for Financial Data")

    # Add a refresh button
    if st.button("Refresh"):
        st.rerun()

    # Dropdown menu to choose analysis mode
    analysis_mode = st.selectbox(
        "Choose Analysis Mode",
        ["Analyze Text", "Analyze PDF"]
    )

    if analysis_mode == "Analyze Text":
        st.header("Input Your Text")
        user_input = st.text_area("Enter your text here:", height=300)

        # Define the specific phrases within the scope of text analysis
        specific_phrases = st.multiselect(
            "Select specific financial phrases to prioritize:",
            options=list(phrase_weights.keys()),
            default=None
        )

        if st.button("Analyze"):
            if user_input:
                st.write("Processing entered text...")
                cleaned_text = clean_text_with_priority(user_input, lexicon_words, specific_phrases, phrase_weights)

                with st.spinner("Analyzing sentiment..."):
                    sentiment_analysis, score = get_sentiment_analysis(cleaned_text, temperature=0.05)
                    st.subheader("Sentiment Analysis")
                    st.write(sentiment_analysis)

                # Create and download outcome PDF
                outcome_pdf = PDF()
                outcome_pdf.add_page()
                outcome_pdf.set_font('Arial', 'B', 16)
                outcome_pdf.multi_cell(0, 10, f"Sentiment Analysis Outcome for Input Text\n")
                outcome_pdf.set_font('Arial', '', 14)
                outcome_pdf.multi_cell(0, 10, f"{sentiment_analysis}\n")
                outcome_pdf.multi_cell(0, 10, f"Sentiment Score: {score}\n")
                outcome_pdf_path = f'outcome_output_text.pdf'
                outcome_pdf.output(outcome_pdf_path)

                with open(outcome_pdf_path, "rb") as file:
                    st.download_button(label=f"Download Sentiment Analysis Outcome", data=file, file_name=outcome_pdf_path)

    elif analysis_mode == "Analyze PDF":
        st.header("Upload PDF Files")
        uploaded_files = st.file_uploader("Or upload PDF files", type="pdf", accept_multiple_files=True)

        all_sentiments = []
        pdf_names = []

        # Select phrases to prioritize
        specific_phrases = st.multiselect(
            "Select specific financial phrases to prioritize:",
            options=list(phrase_weights.keys()),
            default=None
        )

        # Adjust the temperature parameter for the sentiment analysis
        temperature = st.slider("Select temperature for sentiment analysis (lower is more deterministic):", 0.0, 1.0, 0.05)

        st.markdown("---")  # Adds a horizontal divider

        if st.button("Analyze"):
            if uploaded_files:
                for uploaded_file in uploaded_files:
                    st.write(f"Processing {uploaded_file.name}...")
                    pdf_names.append(uploaded_file.name)

                    # Extract text from the uploaded PDF
                    with st.spinner("Extracting text from PDF..."):
                        extracted_text = extract_text_from_pdf(uploaded_file)

                    # Clean the extracted text
                    with st.spinner("Cleaning text..."):
                        cleaned_text = clean_text_with_priority(extracted_text, lexicon_words, specific_phrases, phrase_weights)

                    # Perform sentiment analysis on the cleaned text
                    with st.spinner("Analyzing sentiment..."):
                        sentiment_analysis, score = get_sentiment_analysis(cleaned_text, temperature)
                        all_sentiments.append(score)

                    # Display the conclusion for the current PDF
                    st.subheader(f"Conclusion for {uploaded_file.name}")
                    st.write(sentiment_analysis)

                    # Create and download cleaned PDF
                    cleaned_pdf = PDF()
                    cleaned_pdf.add_page()
                    cleaned_pdf.body(cleaned_text)
                    output_pdf_path = f'cleaned_output_{uploaded_file.name}'
                    cleaned_pdf.output(output_pdf_path)

                    with open(output_pdf_path, "rb") as file:
                        st.download_button(label=f"Download Cleaned PDF for {uploaded_file.name}", data=file, file_name=output_pdf_path)

                    # Create and download outcome PDF
                    outcome_pdf = PDF()
                    outcome_pdf.add_page()
                    outcome_pdf.set_font('Arial', 'B', 16)
                    outcome_pdf.multi_cell(0, 10, f"Sentiment Analysis Outcome for {uploaded_file.name}\n")
                    outcome_pdf.set_font('Arial', '', 14)
                    outcome_pdf.multi_cell(0, 10, f"{sentiment_analysis}\n")
                    outcome_pdf.multi_cell(0, 10, f"Sentiment Score: {score}\n")
                    outcome_pdf_path = f'outcome_output_{uploaded_file.name}.pdf'
                    outcome_pdf.output(outcome_pdf_path)

                    with open(outcome_pdf_path, "rb") as file:
                        st.download_button(label=f"Download Sentiment Analysis Outcome for {uploaded_file.name}", data=file, file_name=outcome_pdf_path)

                # Plot the sentiment scores
                sentiment_df = pd.DataFrame({
                    'PDF Name': pdf_names,
                    'Sentiment Score': all_sentiments
                })
                plot_sentiment_scores(sentiment_df)

if __name__ == "__main__":
    main()
