from flask import Flask, render_template, request, redirect, url_for, flash
import os
import fitz
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

app = Flask(__name__)
app.secret_key = "your_secret_key"  # Replace with a strong secret key
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize ChromaDB
client = chromadb.Client()
collection = client.create_collection(name="document11_QA")

def extract_full_text(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text() + "\n"
    doc.close()
    return full_text

def process_text_to_chunks(text_content):
    def token_len(prompt):
        return len(prompt.split())  # Simplified for demonstration

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=100,
        length_function=token_len,
        separators=[".", "!", "?", ";", "\n"]
    )
    return text_splitter.split_text(text_content)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        if file and file.filename.endswith('.pdf'):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            
            # Extract text and process
            text_content = extract_full_text(file_path)
            chunks = process_text_to_chunks(text_content)
            for num, chunk in enumerate(chunks):
                collection.add(documents=[chunk], ids=[f"ChunkId{num}"])
            
            flash('File successfully processed!')
            return redirect(url_for('query', filename=file.filename))
        
        flash('Allowed file type is PDF')
        return redirect(request.url)

    return render_template('index.html')

@app.route('/query', methods=['GET', 'POST'])
def query():
    if request.method == 'POST':
        query_text = request.form['query']
        results = collection.query(query_texts=[query_text], n_results=1)
        return render_template('results.html', query=query_text, results=results["documents"][0])
    
    return render_template('query.html')

if __name__ == '__main__':
    app.run(debug=True)
