import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from rag.embedder import get_vectorstore

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
    Returns answer + source filenames.

    Args:
        question: user's question string
        memory:   ConversationBufferMemory object for this session
    """
    # Guard: check ChromaDB has documents
    vectorstore = get_vectorstore()
    if vectorstore._collection.count() == 0:
        raise ValueError(
            "No documents uploaded yet. Please upload a PDF, DOCX, or TXT file first."
        )

    chain  = build_chain(memory)
    result = chain.invoke({"question": question})

    # Extract sources from returned documents
    sources = list(set(
        doc.metadata.get("source", "unknown")
        for doc in result.get("source_documents", [])
    ))

    return {
        "answer":  result["answer"],
        "sources": sources
    }