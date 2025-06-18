from flask import Flask, request, jsonify
import os
from python_script import (
    extract_full_text,
    split_text_into_chunks,
    add_chunks_to_collection,
    query_chromadb,
    setup_huggingface_pipeline,
    generate_answer,
)

app = Flask(__name__)

hf_pipeline = setup_huggingface_pipeline()

@app.route('/')
def index():
    # return ("aswin")
    return jsonify({
        "message": "Welcome to the PDF Processing API.",
        "endpoints": {
            "/upload_pdf": "POST - Upload a PDF and extract text.",
            "/split_text": "POST - Split extracted text into chunks.",
            "/generate_answer": "POST - Generate an answer for a question."
        }
    }), 200

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    """Handles PDF upload and extracts text."""
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file and file.filename.endswith('.pdf'):
        pdf_path = os.path.join("uploads", file.filename)
        file.save(pdf_path)
        full_text = extract_full_text(pdf_path)
        return jsonify({"extracted_text": full_text}), 200
    return jsonify({"error": "Invalid file type"}), 400

@app.route('/split_text', methods=['POST'])
def split_text():
    """Splits uploaded text into chunks."""
    data = request.json
    full_text = data.get("text", "")
    if not full_text:
        return jsonify({"error": "Text is required"}), 400

    chunks = split_text_into_chunks(full_text)
    add_chunks_to_collection(chunks)
    return jsonify({"chunks": chunks}), 200

@app.route('/generate_answer', methods=['POST'])
def query_answer():
    """Handles question and generates an answer."""
    data = request.json
    question = data.get("question", "")
    if not question:
        return jsonify({"error": "Question is required"}), 400

    context = query_chromadb(question)
    answer = generate_answer(question, hf_pipeline)
    return jsonify({"question": question, "answer": answer, "context": context}), 200

if __name__ == "__main__":
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    print(app.url_map)  
    app.run(debug=True)
