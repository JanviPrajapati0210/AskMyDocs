import os
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter


# 1.Load any file type 

def load_document(file_path: str):
    """
    Accepts a file path, detects its type,
    and returns a list of LangChain Document objects.
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        loader = PyPDFLoader(file_path)

    elif ext == ".docx":
        loader = Docx2txtLoader(file_path)

    elif ext == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")

    else:
        raise ValueError(f"Unsupported file type: {ext}")

    documents = loader.load()
    return documents          

# 2.Split documents into chunks 

def chunk_documents(documents):
    """
    Splits a list of Document objects into smaller overlapping chunks.
    Returns a new list of smaller Document objects.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,       
        chunk_overlap=200,      
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.split_documents(documents)
    return chunks


# 3. Combined helper (load + chunk in one call) 

def load_and_chunk(file_path: str):
    """
    Main function used by the rest of the app.
    Loads a file and returns chunked Document objects ready for embedding.
    """
    documents = load_document(file_path)
    chunks = chunk_documents(documents)

    print(f"Loaded:  {len(documents)} page(s) from {os.path.basename(file_path)}")
    print(f"Chunked: {len(chunks)} chunks created")

    return chunks