import streamlit as st
import os 
import dotenv
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
env = os.getenv("GROQ_API_KEY")
client = Groq(api_key=env)

USER_AGENT = os.getenv(
    "USER_AGENT",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)


# LangChain + FAISS
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from youtube_transcript_api import YouTubeTranscriptApi

# file loaders
from PyPDF2 import PdfReader  
from docx import Document as DocxDocument
from langchain_core.documents import Document

# importing helpers
from helpers.chunker import chunk_data
from helpers.youtubeloader import load_from_youtube
from helpers.vectorstore import create_vector_store
from helpers.retriever import create_retriever
from helpers.chain import create_rag_chain
from helpers.loaddoc import load_all_docs

from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader

st.set_page_config(page_title="Answering Machine")
st.title("Hey, Shoot Your Questions")

# Initialize session state
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

# Sidebar setup
with st.sidebar:
    st.header("Setup")
    input_type = st.selectbox("Input Type", ["Please pick an option","YouTube Link", "PDF", "Text", "DOCX"])
    input_data, youtube_url = None, None

    if input_type == "YouTube Link":
        youtube_url = st.text_input("Enter the YouTube URL:")
        input_data = youtube_url
    elif input_type == "PDF":
        input_data = st.file_uploader("Upload a PDF file", type=["pdf"])
    elif input_type == "Text":
        input_data = st.text_input("Enter the text here:")
    elif input_type == "DOCX":
        input_data = st.file_uploader("Upload a DOCX file", type=["docx", "doc"])

    if st.button("Process Data"):
        # for youtube
        if input_type == "YouTube Link" and youtube_url:
            with st.spinner("Processing video... This may take a few minutes."):
                try:
                    transcript = load_from_youtube(youtube_url)
                    chunks = chunk_data(transcript)
                    vector_store = create_vector_store(chunks)
                    st.session_state.retriever = create_retriever(vector_store)
                    
                    # Create the RAG chain after the retriever is ready
                    st.session_state.rag_chain = create_rag_chain(st.session_state.retriever)
                    
                    st.success("Video processed successfully!")
                except Exception as e:
                    st.error(f"Error occured: {e}")
                
        # for pdf
        elif input_type == "PDF" and input_data:
            with st.spinner("Processing PDF... This may take a few minutes."):
                try:
                    pdf= load_all_docs(input_data)
                    chunks = chunk_data(pdf)
                    vector_store = create_vector_store(chunks)
                    st.session_state.retriever = create_retriever(vector_store)

                    # Create the RAG chain after the retriever is ready
                    st.session_state.rag_chain = create_rag_chain(st.session_state.retriever)
                    
                    st.success("Pdf processed successfully!")
                except Exception as e:
                    st.error(f"Error occured: {e}")
                
        #for text
        elif input_type == "Text" and input_data:
            with st.spinner("Processing Text... This may take a few minutes."):
                try:
                    docs = [Document(page_content=input_data, metadata={"source": "user_input"})]
                    chunks = chunk_data(docs)
                    vector_store = create_vector_store(chunks)
                    st.session_state.retriever = create_retriever(vector_store)

                    # Create the RAG chain after the retriever is ready
                    st.session_state.rag_chain = create_rag_chain(st.session_state.retriever)
                        
                    st.success("Text processed successfully!")
                except Exception as e:
                    st.error(f"Error occured: {e}")
                

        elif input_type == "DOCX" and input_data:
            with st.spinner("Processing Docx file... This may take a few minutes."):
                try:
                    docs= load_all_docs(input_data)
                    chunks = chunk_data(docs)
                    vector_store = create_vector_store(chunks)
                    st.session_state.retriever = create_retriever(vector_store)

                    # Create the RAG chain after the retriever is ready
                    st.session_state.rag_chain = create_rag_chain(st.session_state.retriever)
                    st.success("Docx processed successfully!")
                except Exception as e:
                    st.error(f"Error occured: {e}")
                
        

# --- Main Q&A section ---
st.header("Q&A")
if st.session_state.rag_chain:
    st.info("Ready to answer questions.")
    question = st.text_input("Ask a question:")

    if question:
        with st.spinner("Generating answer..."):
            try:
                answer = st.session_state.rag_chain.invoke(question)
                if isinstance(answer, dict) and "result" in answer:
                    st.write(answer, answer.get("result", str(answer)))
                else:
                    st.write("**Answer:**", str(answer))
            except Exception as e:
                st.error(f"Error: {e}")
else:
    st.info("Please process some data first.")