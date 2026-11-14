
from document_loader import load_pdf

documents = load_pdf("data/documents/sample.pdf")

print("Total Pages:", len(documents))

for i, doc in enumerate(documents):
    print(f"\nPage {i + 1}")
    print("Text Length:", len(doc.page_content))
    print("Content Preview:")
    print(doc.page_content[:300])
    print("-" * 50)
