import fitz
import re
import chromadb
from chromadb.utils import embedding_functions
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import tiktoken
import os

client = chromadb.Client()
collection = client.create_collection(name="document11_QA")

# Tokenizer setup
tokenizer = tiktoken.get_encoding('cl100k_base')

def extract_full_text(pdf_path):
    """Extracts text from a PDF."""
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text() + "\n"
    doc.close()
    return full_text

def token_len(prompt, dev_input={}):
    """Calculate token length."""
    try:
        tokens = tokenizer.encode(prompt, disallowed_special=())
        return len(tokens)
    except Exception as e:
        print(f"Error in token_len: {e}")
        return 0

def split_text_into_chunks(text):
    """Splits text into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2500, chunk_overlap=100, length_function=token_len, separators=[".", "!", "?", ";", "\n"]
    )
    return text_splitter.split_text(text)

def add_chunks_to_collection(text_chunks):
    """Add text chunks to ChromaDB collection."""
    for num, chunk in enumerate(text_chunks):
        collection.add(documents=chunk, ids="ChunkId" + str(num))

def query_chromadb(query):
    """Query the ChromaDB collection."""
    results = collection.query(query_texts=[query], n_results=1)
    context = {"A" + str(num + 1): str(doc) for num, doc in enumerate(results["documents"][0])}
    return context

def setup_huggingface_pipeline():
    """Set up the Hugging Face pipeline."""
    model_id = "gpt2"  # Change as needed
    model = AutoModelForCausalLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100)
    return pipe

def generate_answer(question, pipe):
    """Generate an answer using the Hugging Face pipeline."""
    result = pipe(question, max_length=150)
    return result[0]['generated_text']
