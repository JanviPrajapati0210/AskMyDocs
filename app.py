import os
import traceback
import uuid
from flask import Flask, request, jsonify, render_template, session
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from rag.loader import load_and_chunk
from rag.embedder import add_documents, reset_vectorstore
from rag.chain import ask, create_memory

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev-secret-change-in-production")

#  Config 
UPLOAD_FOLDER      = "uploads"
ALLOWED_EXTENSIONS = {"pdf", "docx", "txt"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

conversation_memories = {}

def allowed_file(filename):
    return "." in filename and \
           filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def get_memory():
    """Gets or creates a memory object for the current session."""
    # Assign a session ID to this browser if it doesn't have one
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())

    sid = session["session_id"]

    # Create fresh memory if this is a new session
    if sid not in conversation_memories:
        conversation_memories[sid] = create_memory()

    return conversation_memories[sid]


#  Route 1: Serve the chat UI 
@app.route("/")
def home():
    return render_template("index.html")


#  Route 2: Upload documents 
@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file attached to the request."}), 400

    files = request.files.getlist("file")
    if not files or all(f.filename == "" for f in files):
        return jsonify({"error": "No files selected."}), 400

    uploaded     = []
    total_chunks = 0

    for file in files:
        if file.filename == "":
            continue
        if not allowed_file(file.filename):
            return jsonify({
                "error": f"'{file.filename}' is not supported. Use PDF, DOCX, or TXT."
            }), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        try:
            chunks = load_and_chunk(filepath)
            if len(chunks) == 0:
                return jsonify({
                    "error": f"'{filename}' appears to be empty or unreadable."
                }), 400
            add_documents(chunks)
            total_chunks += len(chunks)
            uploaded.append(filename)
        except Exception as e:
            return jsonify({
                "error": f"Failed to process '{filename}': {str(e)}"
            }), 500

    return jsonify({
        "status":  "success",
        "files":   uploaded,
        "chunks":  total_chunks,
        "message": f"Ready! Processed {len(uploaded)} file(s) into {total_chunks} searchable chunks."
    })


# Route 3: Ask a question 
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid request — expected JSON body."}), 400

    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "Question cannot be empty."}), 400
    if len(question) > 2000:
        return jsonify({"error": "Question too long (max 2000 characters)."}), 400

    try:
        memory = get_memory()              # get this session's memory
        result = ask(question, memory)     # pass memory into ask()
        return jsonify({
            "answer":  result["answer"],
            "sources": result["sources"]
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "error": "Something went wrong generating the answer. Please try again."
        }), 500


#  Route 4: New chat (clear memory only) 
@app.route("/new-chat", methods=["POST"])
def new_chat():
    """Clears conversation memory for this session — keeps documents."""
    if "session_id" in session:
        sid = session["session_id"]
        conversation_memories[sid] = create_memory()   # fresh memory
    return jsonify({"status": "success", "message": "Conversation cleared."})


#  Route 5: Reset everything 
@app.route("/reset", methods=["POST"])
def reset():
    try:
        reset_vectorstore()
        for f in os.listdir(UPLOAD_FOLDER):
            os.remove(os.path.join(UPLOAD_FOLDER, f))
        # Also clear this session's memory
        if "session_id" in session:
            sid = session["session_id"]
            conversation_memories[sid] = create_memory()
        return jsonify({"status": "success", "message": "All documents and chat cleared."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


#  Run
if __name__ == "__main__":
    app.run(debug=True, port=5000)