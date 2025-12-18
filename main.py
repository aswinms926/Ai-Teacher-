from ingestion.document_loader import DocumentLoader
from ingestion.text_processor import chunk_text

def main():
    loader = DocumentLoader()
    text = loader.load_pdf("sample.pdf")

    chunks = chunk_text(text)

    print("PHASE 2 VERIFICATION")
    print("====================")
    print("Total characters:", len(text))
    print("Total chunks:", len(chunks))

    print("\n--- FIRST CHUNK (preview) ---\n")
    print(chunks[0])

    print("\n--- SECOND CHUNK (preview) ---\n")
    print(chunks[1])

    print("\n--- LAST CHUNK (preview) ---\n")
    print(chunks[-1])

if __name__ == "__main__":
    main()
