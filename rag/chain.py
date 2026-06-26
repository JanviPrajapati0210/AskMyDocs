import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from rag.embedder import get_vectorstore
import json

load_dotenv()

# 1. Custom RAG prompt 
# This is shown to the LLM when generating the final answer
QA_PROMPT = PromptTemplate.from_template("""
You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't have enough information to answer that."
Do NOT make up information. Be concise and clear.

Context:
{context}

Question: {question}

Answer:
""")


#  2. Build a fresh memory object 
def create_memory():
    """
    Creates a new ConversationBufferMemory.
    Called once per user session — stores the full chat history.
    """
    return ConversationBufferMemory(
        memory_key="chat_history",   # key the chain reads history from
        return_messages=True,         # store as Message objects (not plain text)
        output_key="answer"           # which chain output to save to memory
    )


#  3. Build the conversational RAG chain
def build_chain(memory):
    """
    Builds ConversationalRetrievalChain — takes memory as input
    so history is preserved across calls.
    """
    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY")
    )

    vectorstore = get_vectorstore()
    retriever   = vectorstore.as_retriever(search_kwargs={"k": 4})

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
        return_source_documents=True,   # gives us source docs for citations
        output_key="answer",
        verbose=False
    )
    return chain


# 4. Ask a question (main entry point) 


def ask(question: str, memory) -> dict:
    """
    Runs the conversational RAG chain with memory.
    Returns answer + rich source citations (filename, page, excerpt).
    """
    # Guard: check ChromaDB has documents
    vectorstore = get_vectorstore()
    if vectorstore._collection.count() == 0:
        raise ValueError(
            "No documents uploaded yet. Please upload a PDF, DOCX, or TXT file first."
        )

    chain  = build_chain(memory)
    result = chain.invoke({"question": question})

    # Build rich citations
    seen    = set()
    sources = []

    for doc in result.get("source_documents", []):
        # Extract metadata
        raw_source = doc.metadata.get("source", "unknown")
        filename   = os.path.basename(raw_source)   # strip full path
        page       = doc.metadata.get("page", None)  # PDF page number (0-indexed)
        content    = doc.page_content.strip()

        # Build a short excerpt — first 120 characters, clean whitespace
        excerpt = " ".join(content.split())[:120]
        if len(" ".join(content.split())) > 120:
            excerpt += "..."

        # Deduplicate by filename + page combo
        key = f"{filename}_{page}"
        if key in seen:
            continue
        seen.add(key)

        sources.append({
            "filename": filename,
            "page":     (page + 1) if page is not None else None,  # convert to 1-indexed
            "excerpt":  excerpt
        })

    return {
        "answer":  result["answer"],
        "sources": sources           # now a list of dicts, not just strings
    }


def ask_stream(question: str, memory):
    """
    Generator version of ask() — yields tokens one by one for SSE streaming.
    Yields dicts: {"token": "..."} for each chunk, then {"done": True, "sources": [...]}
    """
    # Guard: check ChromaDB has documents
    vectorstore = get_vectorstore()
    if vectorstore._collection.count() == 0:
        yield {"error": "No documents uploaded yet. Please upload a file first."}
        return

    try:
        chain = build_chain(memory)

        # Collect full answer for memory update + sources
        full_answer = ""

        # stream() yields string chunks from the LLM
        for chunk in chain.stream({"question": question}):
            # ConversationalRetrievalChain streams the answer key
            token = chunk.get("answer", "")
            if token:
                full_answer += token
                yield {"token": token}

        # After streaming — get source citations
        source_docs = vectorstore.similarity_search(question, k=4)
        seen    = set()
        sources = []
        for doc in source_docs:
            filename = os.path.basename(doc.metadata.get("source", "unknown"))
            page     = doc.metadata.get("page", None)
            excerpt  = " ".join(doc.page_content.split())[:120] + "..."
            key = f"{filename}_{page}"
            if key in seen:
                continue
            seen.add(key)
            sources.append({
                "filename": filename,
                "page":     (page + 1) if page is not None else None,
                "excerpt":  excerpt
            })

        # Final signal — tell frontend streaming is done
        yield {"done": True, "sources": sources}

    except Exception as e:
        yield {"error": str(e)}