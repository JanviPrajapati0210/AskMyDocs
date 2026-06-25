import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from rag.embedder import get_vectorstore

load_dotenv()

#  1. The prompt template 
RAG_PROMPT = PromptTemplate.from_template("""
You are a helpful assistant. Answer the question using ONLY the context provided below.
If the answer is not in the context, say "I don't have enough information to answer that."
Do NOT make up information. Be concise and clear.

Context:
{context}

Question: {question}

Answer:
""")


# 2. Format retrieved chunks into a single context string
def format_docs(docs):
    """
    Joins retrieved Document objects into one context string.
    Each chunk is numbered and its source is shown.
    """
    formatted = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "unknown")
        formatted.append(f"[{i+1}] (Source: {source})\n{doc.page_content}")
    return "\n\n".join(formatted)


# 3. Build the full RAG chain 
def build_chain():
    """
    Builds and returns a LangChain LCEL chain:
    question → retrieve → format → prompt → LLM → answer string
    """
    # Load the LLM 
    llm = ChatGroq(
        model="llama-3.1-8b-instant",       
        temperature=0,                
        api_key=os.getenv("GROQ_API_KEY")
    )

    # Get the retriever from ChromaDB
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4}         # retrieve top 4 chunks
    )

    # Build the chain using LangChain Expression Language (LCEL)
    chain = (
        {
            "context":  retriever | format_docs,   # retrieve chunks → format
            "question": RunnablePassthrough()       # pass question through unchanged
        }
        | RAG_PROMPT        # inject context + question into prompt template
        | llm               # send prompt to Groq
        | StrOutputParser() # parse LLM response to plain string
    )

    return chain


# 4. Ask a question 

def ask(question: str) -> dict:
    """
    Runs the RAG chain. Returns answer + sources.
    Raises ValueError if no documents have been uploaded yet.
    """
    # Guard: check ChromaDB has documents
    vectorstore = get_vectorstore()
    collection  = vectorstore._collection
    if collection.count() == 0:
        raise ValueError(
            "No documents uploaded yet. Please upload a PDF, DOCX, or TXT file first."
        )

    #  Run chain 
    chain  = build_chain()
    answer = chain.invoke(question)

    #  Get sources 
    source_docs = vectorstore.similarity_search(question, k=4)
    sources = list(set(
        doc.metadata.get("source", "unknown") for doc in source_docs
    ))

    return {"answer": answer, "sources": sources}