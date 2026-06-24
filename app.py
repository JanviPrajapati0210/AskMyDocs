# app.py
import os
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from rag.loader import load_and_chunk
from rag.embedder import add_documents, reset_vectorstore
from rag.chain import ask

load_dotenv()

app = Flask(__name__)

# ── Config ─────────────────────────────────────────────────────────────────
UPLOAD_FOLDER   = "uploads"
ALLOWED_EXTENSIONS = {"pdf", "docx", "txt"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return "." in filename and \
           filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ── Route 1: Serve the chat UI ─────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("index.html")


# ── Route 2: Upload documents ──────────────────────────────────────────────
@app.route("/upload", methods=["POST"])
def upload():
    # Validation
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    files = request.files.getlist("file")   # support multiple files
    if not files:
        return jsonify({"error": "No files selected"}), 400

    uploaded = []
    total_chunks = 0

    for file in files:
        if file.filename == "":
            continue
        if not allowed_file(file.filename):
            return jsonify({"error": f"Unsupported file: {file.filename}"}), 400

        # Save file securely
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        # Process: load → chunk → store in ChromaDB
        try:
            chunks = load_and_chunk(filepath)
            add_documents(chunks)
            total_chunks += len(chunks)
            uploaded.append(filename)
        except Exception as e:
            return jsonify({"error": f"Failed to process {filename}: {str(e)}"}), 500

    return jsonify({
        "status": "success",
        "files": uploaded,
        "chunks": total_chunks,
        "message": f"Processed {len(uploaded)} file(s) into {total_chunks} chunks"
    })


# ── Route 3: Ask a question ────────────────────────────────────────────────
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "Question cannot be empty"}), 400

    try:
        result = ask(question)
        return jsonify({
            "answer":  result["answer"],
            "sources": result["sources"]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Route 4: Reset (clear all documents) ──────────────────────────────────
@app.route("/reset", methods=["POST"])
def reset():
    try:
        reset_vectorstore()
        # Also clear uploads folder
        for f in os.listdir(UPLOAD_FOLDER):
            os.remove(os.path.join(UPLOAD_FOLDER, f))
        return jsonify({"status": "success", "message": "All documents cleared"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Run ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=5000)