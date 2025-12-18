# PDF Document Loader - Usage Guide

## Overview

The `DocumentLoader` class in `ingestion/document_loader.py` provides functionality to extract text from PDF files for use in the RAG-based AI Tutor system.

## Features

✅ **Multi-page PDF support** - Extracts text from all pages  
✅ **Error handling** - Graceful handling of corrupted or invalid PDFs  
✅ **Logging** - Detailed logging of extraction process  
✅ **Text cleaning** - Removes excessive whitespace and normalizes text  
✅ **Path validation** - Validates file existence and format  

## Installation

Install the required dependency:

```bash
pip install PyPDF2
```

Or install all project dependencies:

```bash
pip install -r requirements.txt
```

## Basic Usage

### Example 1: Simple PDF Loading

```python
from ingestion.document_loader import DocumentLoader

# Initialize the loader
loader = DocumentLoader()

# Load a PDF file
text = loader.load_pdf("path/to/your/document.pdf")

# Use the extracted text
print(f"Extracted {len(text)} characters")
print(text[:500])  # Preview first 500 characters
```

### Example 2: With Error Handling

```python
from ingestion.document_loader import DocumentLoader

loader = DocumentLoader()

try:
    text = loader.load_pdf("educational_material.pdf")
    
    if text:
        print(f"✓ Successfully extracted {len(text)} characters")
        # Process the text further...
    else:
        print("⚠ PDF was empty or no text could be extracted")
        
except FileNotFoundError:
    print("✗ PDF file not found")
    
except ValueError as e:
    print(f"✗ Invalid file or missing dependency: {e}")
    
except Exception as e:
    print(f"✗ Error loading PDF: {e}")
```

### Example 3: Processing Multiple PDFs

```python
from ingestion.document_loader import DocumentLoader
from pathlib import Path

loader = DocumentLoader()

# Get all PDF files in a directory
pdf_dir = Path("./educational_materials")
pdf_files = list(pdf_dir.glob("*.pdf"))

# Process each PDF
all_documents = {}

for pdf_file in pdf_files:
    try:
        text = loader.load_pdf(str(pdf_file))
        all_documents[pdf_file.name] = text
        print(f"✓ Loaded: {pdf_file.name}")
    except Exception as e:
        print(f"✗ Failed to load {pdf_file.name}: {e}")

print(f"\nTotal documents loaded: {len(all_documents)}")
```

## Testing

Use the provided test script to verify PDF loading:

```bash
# Test with a specific PDF file
python test_pdf_loader.py path/to/sample.pdf

# Example output:
# ============================================================
# PDF Document Loader Test
# ============================================================
# 
# Initializing DocumentLoader...
# 
# Loading PDF: sample.pdf
# ------------------------------------------------------------
# 
# ✓ PDF loaded successfully!
# 
# Total characters extracted: 15234
# Total words (approx): 2456
# Total lines: 342
```

## How It Works

### Step-by-Step Process

1. **Validation**
   - Checks if PyPDF2 is installed
   - Validates file path exists
   - Confirms file has `.pdf` extension

2. **PDF Reading**
   - Opens PDF using PyPDF2's `PdfReader`
   - Counts total number of pages
   - Logs page count for tracking

3. **Text Extraction**
   - Iterates through each page
   - Extracts text using `extract_text()`
   - Handles page-level errors gracefully
   - Continues processing even if some pages fail

4. **Text Combination**
   - Joins all page texts with double newlines
   - Preserves paragraph structure

5. **Text Cleaning**
   - Removes null characters (`\x00`)
   - Normalizes multiple spaces to single space
   - Normalizes multiple newlines to double newline
   - Strips leading/trailing whitespace

6. **Return**
   - Returns cleaned text as a single string
   - Logs total characters extracted

## Logging

The loader uses Python's built-in logging module. Configure logging level:

```python
import logging

# Set to DEBUG for detailed page-by-page extraction info
logging.basicConfig(level=logging.DEBUG)

# Set to INFO for general progress (default)
logging.basicConfig(level=logging.INFO)

# Set to WARNING to only see warnings and errors
logging.basicConfig(level=logging.WARNING)
```

### Log Messages

- **INFO**: Initialization, file opening, page count, success summary
- **DEBUG**: Per-page extraction details
- **WARNING**: Empty pages, missing PyPDF2
- **ERROR**: File not found, invalid format, extraction errors

## Error Handling

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `ValueError: PyPDF2 is not installed` | PyPDF2 not installed | Run `pip install PyPDF2` |
| `FileNotFoundError` | PDF file doesn't exist | Check file path |
| `ValueError: File is not a PDF` | Wrong file extension | Ensure file ends with `.pdf` |
| `Exception: Error reading PDF` | Corrupted or encrypted PDF | Try different PDF or decrypt it |

## Limitations

⚠️ **Current Limitations:**

- Does not handle encrypted/password-protected PDFs
- Does not extract images or tables
- Does not preserve formatting (bold, italic, etc.)
- Does not handle scanned PDFs (OCR not implemented)
- Text chunking is NOT performed (handled by `text_processor.py`)

## Next Steps

After loading PDF text, you should:

1. **Chunk the text** using `ingestion/text_processor.py`
2. **Generate embeddings** using `vector_store/embeddings.py`
3. **Store in vector database** using `vector_store/database.py`

## API Reference

### `DocumentLoader`

#### `__init__()`
Initialize the document loader and check for dependencies.

#### `load_pdf(file_path: str) -> Optional[str]`
Load and extract text from a PDF file.

**Parameters:**
- `file_path` (str): Path to PDF file (absolute or relative)

**Returns:**
- `str`: Extracted and cleaned text, or empty string if no text found

**Raises:**
- `FileNotFoundError`: If PDF file doesn't exist
- `ValueError`: If PyPDF2 not installed or file is not a PDF
- `Exception`: For other PDF reading errors

#### `_clean_text(text: str) -> str`
Internal method to clean extracted text.

**Parameters:**
- `text` (str): Raw extracted text

**Returns:**
- `str`: Cleaned text

## Examples in the Wild

### Integration with RAG Pipeline

```python
from ingestion.document_loader import DocumentLoader
from ingestion.text_processor import TextProcessor
from vector_store.embeddings import EmbeddingGenerator
from vector_store.database import VectorDatabase

# Step 1: Load PDF
loader = DocumentLoader()
text = loader.load_pdf("textbook_chapter1.pdf")

# Step 2: Chunk text (TODO: implement)
processor = TextProcessor(chunk_size=500, chunk_overlap=50)
chunks = processor.chunk_text(text)

# Step 3: Generate embeddings (TODO: implement)
embedder = EmbeddingGenerator()
embeddings = embedder.generate_embeddings_batch(chunks)

# Step 4: Store in vector database (TODO: implement)
db = VectorDatabase()
db.add_documents(chunks, embeddings)
```

---

**Status**: ✅ Fully implemented and ready to use!
