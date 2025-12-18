from ingestion.document_loader import DocumentLoader

if __name__ == "__main__":
    loader = DocumentLoader()

    pdf_path = "sample.pdf"  # put a real PDF here
    text = loader.load_pdf(pdf_path)

    print("PDF loaded successfully!")
    print("Number of characters:", len(text))
    print("\n--- Preview (first 500 chars) ---\n")
    print(text[:500])
