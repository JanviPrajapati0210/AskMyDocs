# test_embedder.py
from rag.loader import load_and_chunk
from rag.embedder import add_documents, search, reset_vectorstore

# Start fresh
reset_vectorstore()

# Create a sample document
with open("test_doc.txt", "w") as f:
    f.write("""
Employees get 18 days of paid annual leave per year.
Leave requests must be submitted 2 weeks in advance through the HR portal.
Medical leave requires a valid doctor's certificate within 48 hours.
Remote work is allowed up to 3 days per week with manager approval.
Office is open Monday to Friday 9am to 6pm.
Python and Flask are used for backend development.
All code must pass review before merging to main branch.
""")

# Load, chunk, and store
chunks = load_and_chunk("test_doc.txt")
add_documents(chunks)

# Now search!
print("\n--- Test Search ---")
query = "How many leave days do I get?"
results = search(query, k=3)

for doc, score in results:
    print(f"\nScore: {score:.4f}")          # lower = more relevant
    print(f"Text:  {doc.page_content[:80]}")
    print(f"Source: {doc.metadata.get('source')}")