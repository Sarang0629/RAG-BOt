import streamlit as st
import re
from PyPDF2 import PdfReader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
import os
import configparser

# Set page layout to wide mode
st.set_page_config(layout="wide")

# Read API key from credentials.ini file
config = configparser.ConfigParser()
config.read('credentials.ini')

# Get the API key
api_key = config.get('API_KEY', 'google_api_key')

# Configure the generative AI model with the API key
os.environ["GOOGLE_API_KEY"] = api_key
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
model = genai.GenerativeModel('gemini-pro')

# Define chat prompt template for querying documents
chat_prompt_template = ChatPromptTemplate.from_messages([
    ("system", "Hello! How can I assist you today?"),
    ("user", "{query}")
])

# RecursiveCharacterTextSplitter Class
class RecursiveCharacterTextSplitter:
    def __init__(self, chunk_size=2000, chunk_overlap=100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, context):
        return [context[i:i + self.chunk_size] for i in range(0, len(context), self.chunk_size - self.chunk_overlap)]

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    try:
        pdf_reader = PdfReader(pdf_file)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    except Exception as e:
        st.error(f"Error processing PDF file: {str(e)}")
    return text

# Function to clean text by removing special characters
def clean_text(text):
    text = re.sub(r'[^A-Za-z0-9\s]+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Function to process text from PDF files
def process_documents(files):
    texts = []
    for file in files:
        raw_text = extract_text_from_pdf(file)
        cleaned_text = clean_text(raw_text)
        texts.append(cleaned_text)

    if texts:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
        combined_text = '\n'.join(texts)
        split_texts = text_splitter.split_text(combined_text)
        return split_texts
    return []

# Function to generate response using the Generative AI model
def generate_response(prompt):
    if 'vector_index' in st.session_state:
        vector_index = st.session_state.vector_index
        docs = vector_index.invoke(prompt)
        retrieved_texts = [doc.page_content for doc in docs]
        context = f"User: {prompt}\nDocuments: {' '.join(retrieved_texts)}"
        chat_prompt = chat_prompt_template.invoke({"query": context})
        prompt_text = chat_prompt.to_string()
        response = model.generate_content(prompt_text)

        response_text = response.text.split("AI:")[1].strip() if "AI:" in response.text else response.text.strip()
        st.markdown(f"**Bot:** {response_text}")
    else:
        st.error("Please process the documents first.")

# Streamlit App
st.title("RAG BOT")
st.write("Upload PDF files to extract, clean, and process the text.")

# Single uploader for multiple PDF files
uploaded_files = st.file_uploader("Choose PDF files", accept_multiple_files=True, type=['pdf'])

# Process button
if st.button("Process Documents"):
    if uploaded_files:
        processed_texts = process_documents(uploaded_files)
        # Embedding creation for indexing
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_index = Chroma.from_texts(processed_texts, embeddings).as_retriever()
        st.session_state.vector_index = vector_index
        st.success("Documents processed and indexed successfully.")
    else:
        st.warning("Please upload PDF files.")

# Define default questions
default_questions = [
    "What is the summary of the documents?",
    "What are the bullet points of the documents?",
    "What are the key takeaways?",
]

# Arrange default questions next to each other
cols = st.columns(len(default_questions))

# Create a callback function to handle default question buttons
def handle_button_click(question):
    st.session_state.selected_question = question
    generate_response(question)

# Display default questions as buttons next to each other
for i, question in enumerate(default_questions):
    with cols[i]:
        if st.button(question):
            handle_button_click(question)

# Allow custom query input
custom_query = st.text_input("Ask a Query", key="custom_query")

if st.button("Get Response", key="get_response"):
    if custom_query:
        generate_response(custom_query)
    else:
        st.warning("Please enter a query.")
