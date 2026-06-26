# test_loader.py — run this to verify loader.py works
from rag.loader import load_and_chunk

# Test with a simple txt file first
with open("test_doc.txt", "w") as f:
    f.write("""
Employees at the company are entitled to 18 days of paid annual leave per year.
Leave requests must be submitted at least two weeks in advance through the HR portal.
Medical leave requires a valid doctor's certificate submitted within 48 hours.
Remote work is allowed up to three days per week with manager approval.
The office operates Monday through Friday from 9am to 6pm.
All code changes must pass peer review before merging to the main branch.
The CI pipeline runs automated tests on every pull request to ensure quality.
Python and Flask are used for the backend API development.
""")

chunks = load_and_chunk("test_doc.txt")

print(f"\nTotal chunks: {len(chunks)}")
print(f"\n--- Chunk 1 ---")
print(chunks[0].page_content)
print(f"\n--- Chunk 1 metadata ---")
print(chunks[0].metadata)