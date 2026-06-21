import os
import chromadb
from chromadb.config import Settings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


CHROMA_DIR   = "chroma_db"          
COLLECTION   = "rag_documents"      
EMBED_MODEL  = "all-MiniLM-L6-v2"  
TOP_K        = 4                    


# 1. Load the embedding model (done once, reused) 

def get_embeddings():
    """Returns a LangChain-compatible HuggingFace embedding model."""
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},   
        encode_kwargs={"normalize_embeddings": True}
    )


#  2. Get (or create) the ChromaDB vector store 

def get_vectorstore():
    """
    Opens existing ChromaDB collection from disk,
    or creates a new one if it doesn't exist yet.
    """
    embeddings = get_embeddings()
    vectorstore = Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,   
    )
    return vectorstore


#  3. Add new document chunks to ChromaDB 

def add_documents(chunks):
    """
    Embeds a list of LangChain Document chunks and stores
    them in ChromaDB. Safe to call multiple times —
    ChromaDB appends without wiping existing data.
    """
    vectorstore = get_vectorstore()
    vectorstore.add_documents(chunks)
    print(f"Added {len(chunks)} chunks to ChromaDB")
    return vectorstore


#  4. Search ChromaDB with a user query 

def search(query: str, k: int = TOP_K):
    """
    Converts query to embedding, finds top-k most similar chunks.
    Returns list of (Document, score) tuples.
    """
    vectorstore = get_vectorstore()
    results = vectorstore.similarity_search_with_score(query, k=k)
    return results


#  5. Reset ChromaDB (clears everything — use carefully) 

def reset_vectorstore():
    """Wipes the ChromaDB collection. Used when user re-uploads all docs."""
    import shutil
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)
        print("ChromaDB cleared.")