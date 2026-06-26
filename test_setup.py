# test_setup.py
from dotenv import load_dotenv
import os
load_dotenv()

import langchain
print(f"LangChain:           OK ({langchain.__version__})")

import chromadb
print(f"ChromaDB:            OK ({chromadb.__version__})")

from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
vec = model.encode("hello world")
print(f"SentenceTransformer: OK (vector size: {len(vec)})")

from langchain_groq import ChatGroq
groq_key = os.getenv("GROQ_API_KEY")
print(f"Groq API key:        {'OK' if groq_key else 'MISSING'}")

import flask
print(f"Flask:               OK ({flask.__version__})")