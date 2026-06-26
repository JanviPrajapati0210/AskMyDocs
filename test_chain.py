# test_chain.py
from rag.loader import load_and_chunk
from rag.embedder import add_documents, reset_vectorstore
from rag.chain import ask

# Fresh start
reset_vectorstore()

# Load a document
with open("test_doc.txt", "w") as f:
    f.write("""
Employees get 18 days of paid annual leave per year.
Leave requests must be submitted 2 weeks in advance through the HR portal.
Medical leave requires a valid doctor's certificate within 48 hours.
Remote work is allowed up to 3 days per week with manager approval.
Office is open Monday to Friday 9am to 6pm.
Python and Flask are used for backend development.
All code must pass code review before merging to the main branch.
""")

chunks = load_and_chunk("test_doc.txt")
add_documents(chunks)

# Test questions
questions = [
    "How many leave days do employees get?",
    "Can I work from home?",
    "What happens if I need medical leave?",
    "What is the capital of France?",   # should say "I don't have enough info"
]

for q in questions:
    print(f"\n{'='*50}")
    print(f"Q: {q}")
    result = ask(q)
    print(f"A: {result['answer']}")
    print(f"Sources: {result['sources']}")