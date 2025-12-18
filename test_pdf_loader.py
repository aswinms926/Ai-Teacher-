"""
Test Script for PDF Document Loader

This script demonstrates how to use the DocumentLoader to extract text from PDFs.

Usage:
    python test_pdf_loader.py <path_to_pdf_file>
"""

import sys
from pathlib import Path

# Add parent directory to path to import our modules
sys.path.insert(0, str(Path(__file__).parent))

from ingestion.document_loader import DocumentLoader


def test_pdf_loading(pdf_path: str):
    """
    Test the PDF loading functionality.
    
    Args:
        pdf_path: Path to a PDF file to test
    """
    print("=" * 60)
    print("PDF Document Loader Test")
    print("=" * 60)
    print()
    
    # Initialize the document loader
    print("Initializing DocumentLoader...")
    loader = DocumentLoader()
    print()
    
    # Load the PDF
    print(f"Loading PDF: {pdf_path}")
    print("-" * 60)
    
    try:
        # Extract text from PDF
        text = loader.load_pdf(pdf_path)
        
        # Display results
        print()
        print("✓ PDF loaded successfully!")
        print()
        print(f"Total characters extracted: {len(text)}")
        print(f"Total words (approx): {len(text.split())}")
        print(f"Total lines: {len(text.splitlines())}")
        print()
        
        # Show preview of extracted text
        print("=" * 60)
        print("Text Preview (first 500 characters):")
        print("=" * 60)
        print(text[:500])
        
        if len(text) > 500:
            print("\n... (truncated)")
        
        print()
        print("=" * 60)
        print("Text Preview (last 500 characters):")
        print("=" * 60)
        print(text[-500:])
        print()
        
        return text
        
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        print("\nPlease provide a valid PDF file path.")
        
    except ValueError as e:
        print(f"✗ Error: {e}")
        print("\nMake sure PyPDF2 is installed: pip install PyPDF2")
        
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        
    return None


def main():
    """Main function to run the test."""
    
    # Check if PDF path is provided
    if len(sys.argv) < 2:
        print("Usage: python test_pdf_loader.py <path_to_pdf_file>")
        print()
        print("Example:")
        print("  python test_pdf_loader.py sample.pdf")
        print("  python test_pdf_loader.py C:\\Documents\\textbook.pdf")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    # Run the test
    result = test_pdf_loading(pdf_path)
    
    if result:
        print("=" * 60)
        print("✓ Test completed successfully!")
        print("=" * 60)
    else:
        print("=" * 60)
        print("✗ Test failed. See errors above.")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
