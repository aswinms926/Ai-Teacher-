# Text Processor - Usage Guide

## Overview

The `TextProcessor` class in `ingestion/text_processor.py` provides intelligent text preprocessing and chunking for RAG systems. It cleans text, splits it into semantic chunks, and preserves context across boundaries.

## Why Chunking is Needed for RAG

**RAG (Retrieval-Augmented Generation)** systems work by:
1. Breaking documents into chunks
2. Converting chunks to embeddings (vectors)
3. Storing embeddings in a vector database
4. Retrieving relevant chunks based on user queries
5. Using retrieved chunks as context for LLM responses

**Why we need chunking:**
- ✅ Embedding models have token limits (512-8192 tokens)
- ✅ Smaller chunks = more precise retrieval
- ✅ Larger chunks = more context but less precision
- ✅ Optimal size balances precision vs. context

## How Chunk Size Affects Retrieval Quality

| Chunk Size | Pros | Cons | Use Case |
|------------|------|------|----------|
| **< 200 chars** | Very precise | Lacks context | Keywords, definitions |
| **500-800 chars** ✅ | Good balance | - | Most applications |
| **> 1500 chars** | Rich context | Less precise | Long-form content |

**Overlap between chunks** preserves context across boundaries and prevents information loss.

## Installation

No additional dependencies required beyond standard Python libraries (`re`, `logging`, `typing`).

## Basic Usage

### Example 1: Simple Text Chunking

```python
from ingestion.text_processor import TextProcessor

# Initialize with default settings (500 chars, 50 overlap)
processor = TextProcessor()

# Chunk some text
text = """
Your long document here...
Multiple paragraphs...
"""

chunks = processor.chunk_text(text)

print(f"Created {len(chunks)} chunks")
for i, chunk in enumerate(chunks, 1):
    print(f"Chunk {i}: {len(chunk)} characters")
```

### Example 2: Custom Chunk Size

```python
from ingestion.text_processor import TextProcessor

# Larger chunks for more context
processor = TextProcessor(chunk_size=800, chunk_overlap=100)

chunks = processor.chunk_text(long_document)
```

### Example 3: Text Cleaning Only

```python
from ingestion.text_processor import TextProcessor

processor = TextProcessor()

dirty_text = "Hello....    world!!!   Too   many   spaces."
clean_text = processor.clean_text(dirty_text)

print(clean_text)  # "Hello. world! Too many spaces."
```

### Example 4: Extract Metadata

```python
from ingestion.text_processor import TextProcessor

processor = TextProcessor()

metadata = processor.extract_metadata(document)

print(f"Characters: {metadata['char_count']}")
print(f"Words: {metadata['word_count']}")
print(f"Paragraphs: {metadata['paragraph_count']}")
print(f"Headers: {metadata['potential_headers']}")
```

## Complete RAG Pipeline Example

```python
from ingestion.document_loader import DocumentLoader
from ingestion.text_processor import TextProcessor

# Step 1: Load PDF
loader = DocumentLoader()
raw_text = loader.load_pdf("textbook.pdf")

# Step 2: Process and chunk text
processor = TextProcessor(chunk_size=600, chunk_overlap=60)
chunks = processor.chunk_text(raw_text)

print(f"Processed {len(chunks)} chunks from PDF")

# Step 3: Use chunks for embeddings (next phase)
# embeddings = generate_embeddings(chunks)
# store_in_vector_db(chunks, embeddings)
```

## Features

### Text Cleaning

The `clean_text()` method performs:

1. **Remove control characters** - Null bytes, control codes
2. **Normalize unicode** - Smart quotes → regular quotes
3. **Fix excessive punctuation** - `....` → `...`, `!!!` → `!`
4. **Normalize whitespace** - Multiple spaces → single space
5. **Fix line breaks** - Multiple newlines → double newline
6. **Fix punctuation spacing** - Ensure space after periods
7. **Trim lines** - Remove leading/trailing whitespace

**Example:**
```python
processor.clean_text("Hello....    world!!!")
# Returns: "Hello. world!"
```

### Intelligent Chunking

The `chunk_text()` method uses a **multi-level strategy**:

1. **Paragraph-level splitting** (preserves semantic units)
2. **Sentence-level splitting** (for large paragraphs)
3. **Word-boundary overlap** (preserves context)
4. **Post-processing** (merges tiny chunks, removes noise)

**Example:**
```python
chunks = processor.chunk_text(document)
# Returns: ['Chunk 1...', 'Chunk 2...', ...]
```

### Metadata Extraction

The `extract_metadata()` method extracts:

- Character count
- Word count
- Line count
- Paragraph count
- Potential headers (short lines without punctuation)

**Example:**
```python
metadata = processor.extract_metadata(text)
# Returns: {'char_count': 1500, 'word_count': 250, ...}
```

## Configuration

### Recommended Settings

**For general educational content:**
```python
TextProcessor(chunk_size=600, chunk_overlap=60)
```

**For technical documentation:**
```python
TextProcessor(chunk_size=800, chunk_overlap=80)
```

**For Q&A or definitions:**
```python
TextProcessor(chunk_size=300, chunk_overlap=30)
```

**For long-form articles:**
```python
TextProcessor(chunk_size=1000, chunk_overlap=100)
```

### Parameter Guidelines

**chunk_size:**
- Minimum: 100 characters (warning issued)
- Recommended: 500-800 characters
- Maximum: 2000 characters (warning issued)

**chunk_overlap:**
- Recommended: 10-20% of chunk_size
- Must be less than chunk_size
- Must be non-negative

## Logging

The processor uses Python's logging module:

```python
import logging

# Set to DEBUG for detailed chunking info
logging.basicConfig(level=logging.DEBUG)

# Set to INFO for general progress (default)
logging.basicConfig(level=logging.INFO)

# Set to WARNING for warnings only
logging.basicConfig(level=logging.WARNING)
```

**Log messages include:**
- **INFO**: Initialization, chunking start/complete, statistics
- **DEBUG**: Per-paragraph/sentence processing, chunk merging
- **WARNING**: Empty text, very small/large chunk sizes

## Testing

Run the comprehensive test suite:

```bash
python test_text_processor.py
```

**Tests include:**
- Text cleaning (punctuation, whitespace, unicode)
- Chunking with different sizes
- Metadata extraction
- Edge cases (empty text, very long sentences)
- Overlap functionality

## How It Works

### Chunking Algorithm

```
1. Clean the input text
   ↓
2. Split into paragraphs (by \n\n)
   ↓
3. For each paragraph:
   - If small enough → add to current chunk
   - If too large → split by sentences
   ↓
4. When chunk is full:
   - Save current chunk
   - Create overlap from end of chunk
   - Start new chunk with overlap
   ↓
5. Post-process:
   - Remove tiny chunks (< 50 chars)
   - Merge small chunks if possible
   ↓
6. Return list of chunks
```

### Overlap Mechanism

```
Chunk 1: "...context at the end."
         └─────────────┬─────────────┘
                   (overlap)
                       ↓
Chunk 2: "context at the end. New content..."
```

This ensures that information at chunk boundaries isn't lost during retrieval.

## API Reference

### `TextProcessor`

#### `__init__(chunk_size=500, chunk_overlap=50)`
Initialize the text processor.

**Parameters:**
- `chunk_size` (int): Target chunk size in characters (default: 500)
- `chunk_overlap` (int): Overlap between chunks (default: 50)

**Raises:**
- `ValueError`: If overlap >= chunk_size or overlap < 0

---

#### `chunk_text(text: str) -> List[str]`
Split text into semantic chunks.

**Parameters:**
- `text` (str): Input text to chunk

**Returns:**
- `List[str]`: List of text chunks

**Example:**
```python
chunks = processor.chunk_text(document)
```

---

#### `clean_text(text: str) -> str`
Clean and normalize text.

**Parameters:**
- `text` (str): Input text to clean

**Returns:**
- `str`: Cleaned text

**Example:**
```python
clean = processor.clean_text("Messy....  text!!!")
```

---

#### `extract_metadata(text: str) -> Dict[str, any]`
Extract metadata from text.

**Parameters:**
- `text` (str): Input text

**Returns:**
- `Dict`: Metadata dictionary with keys:
  - `char_count`: Total characters
  - `word_count`: Total words
  - `line_count`: Total lines
  - `paragraph_count`: Total paragraphs
  - `potential_headers`: List of potential headers

**Example:**
```python
metadata = processor.extract_metadata(text)
print(metadata['word_count'])
```

## Performance Considerations

**Memory:**
- Processes entire text in memory
- Chunk list stored in memory
- Suitable for documents up to ~10MB

**Speed:**
- Regex-based cleaning is fast
- Chunking is O(n) where n = text length
- Typical speed: ~1MB/second

**Optimization tips:**
- Use appropriate chunk size (don't go too small)
- Process documents in batches if dealing with many files
- Consider streaming for very large documents (> 10MB)

## Limitations

⚠️ **Current Limitations:**

- Does not handle tables or structured data specially
- Does not preserve markdown/HTML formatting
- Sentence splitting is basic (may fail on complex abbreviations)
- Does not detect semantic topic boundaries
- No language-specific processing (English-optimized)

## Next Steps

After chunking, you should:

1. **Generate embeddings** using `vector_store/embeddings.py`
2. **Store in vector database** using `vector_store/database.py`
3. **Retrieve relevant chunks** for user queries
4. **Use in RAG pipeline** for response generation

---

**Status**: ✅ Fully implemented and tested  
**Dependencies**: None (uses Python standard library)  
**Performance**: Fast, suitable for real-time processing
