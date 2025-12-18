# Embeddings Generator - Usage Guide

## Overview

The `EmbeddingGenerator` class in `vector_store/embeddings.py` generates vector embeddings for text chunks using pre-trained API models (OpenAI or Gemini).

## What Are Embeddings?

**Embeddings** are numerical vector representations of text that capture semantic meaning:

- **Vector representation**: Text → Array of numbers (e.g., [0.23, -0.45, 0.67, ...])
- **High-dimensional**: Typically 384-1536 dimensions
- **Semantic similarity**: Similar texts have similar vectors
- **Distance-based**: Vector distance = semantic similarity

### Example:

```
"cat" → [0.2, 0.5, -0.3, 0.1, ...]
"dog" → [0.3, 0.4, -0.2, 0.2, ...]  ← Close to "cat"
"car" → [-0.5, 0.1, 0.8, -0.4, ...] ← Far from "cat"
```

## Why Embeddings Are Needed for RAG

**RAG (Retrieval-Augmented Generation)** relies on embeddings for semantic search:

### Traditional Keyword Search:
```
Query: "What is ML?"
❌ Won't match: "Machine learning is..."
✓ Only matches: "ML is..."
```

### Embedding-Based Semantic Search:
```
Query: "What is ML?" → [0.3, 0.5, ...]
✓ Matches: "Machine learning is..." → [0.31, 0.48, ...]
✓ Matches: "ML is..." → [0.29, 0.52, ...]
```

### How It Works in RAG:

1. **Ingestion Phase**:
   ```
   Document → Chunks → Embeddings → Vector Database
   ```

2. **Retrieval Phase**:
   ```
   User Query → Embedding → Search Vector DB → Top-K Chunks
   ```

3. **Generation Phase**:
   ```
   Query + Retrieved Chunks → LLM → Response
   ```

## Installation

Install required dependencies:

```bash
# For OpenAI
pip install openai

# For Gemini
pip install google-generativeai

# Or install all dependencies
pip install -r requirements.txt
```

## Setup

### Set API Keys

**Option 1: Environment Variables (Recommended)**

```bash
# Windows PowerShell
$env:OPENAI_API_KEY="sk-your-openai-key-here"
$env:GEMINI_API_KEY="your-gemini-key-here"

# Linux/Mac
export OPENAI_API_KEY="sk-your-openai-key-here"
export GEMINI_API_KEY="your-gemini-key-here"
```

**Option 2: .env File**

Create `.env` file in project root:
```
OPENAI_API_KEY=sk-your-openai-key-here
GEMINI_API_KEY=your-gemini-key-here
```

## Basic Usage

### Example 1: OpenAI Embeddings

```python
from vector_store.embeddings import EmbeddingGenerator

# Initialize with OpenAI
generator = EmbeddingGenerator(provider="openai")

# Generate single embedding
text = "Machine learning is a subset of AI."
embedding = generator.generate_embedding(text)

print(f"Embedding dimension: {len(embedding)}")  # 1536
print(f"First 5 values: {embedding[:5]}")
```

### Example 2: Gemini Embeddings

```python
from vector_store.embeddings import EmbeddingGenerator

# Initialize with Gemini
generator = EmbeddingGenerator(provider="gemini")

# Generate single embedding
text = "Biology is the study of life."
embedding = generator.generate_embedding(text)

print(f"Embedding dimension: {len(embedding)}")  # 768
```

### Example 3: Batch Processing

```python
from vector_store.embeddings import EmbeddingGenerator

generator = EmbeddingGenerator(provider="openai")

# Generate embeddings for multiple chunks
chunks = [
    "Photosynthesis converts light to energy.",
    "Cells are the basic unit of life.",
    "DNA stores genetic information."
]

embeddings = generator.generate_embeddings_batch(chunks)

print(f"Generated {len(embeddings)} embeddings")
# Output: Generated 3 embeddings
```

### Example 4: Convenience Function

```python
from vector_store.embeddings import generate_embeddings

# Quick one-liner
embeddings = generate_embeddings(
    ["Text 1", "Text 2", "Text 3"],
    provider="openai"
)
```

## Complete RAG Pipeline Example

```python
from ingestion.document_loader import DocumentLoader
from ingestion.text_processor import chunk_text
from vector_store.embeddings import EmbeddingGenerator

# Step 1: Load PDF
loader = DocumentLoader()
text = loader.load_pdf("biology_textbook.pdf")

# Step 2: Chunk text
chunks = chunk_text(text, chunk_size=600, chunk_overlap=60)

# Step 3: Generate embeddings
generator = EmbeddingGenerator(provider="openai")
embeddings = generator.generate_embeddings_batch(chunks)

print(f"✓ Processed {len(chunks)} chunks")
print(f"✓ Generated {len(embeddings)} embeddings")
print(f"✓ Ready for vector database storage!")
```

## Supported Providers

### OpenAI

**Models:**
- `text-embedding-3-small` (default) - 1536 dimensions, cost-effective
- `text-embedding-3-large` - 3072 dimensions, higher quality
- `text-embedding-ada-002` - 1536 dimensions, legacy

**Pricing (as of 2024):**
- text-embedding-3-small: $0.02 / 1M tokens
- text-embedding-3-large: $0.13 / 1M tokens

**Usage:**
```python
generator = EmbeddingGenerator(
    provider="openai",
    model_name="text-embedding-3-small"
)
```

### Google Gemini

**Models:**
- `text-embedding-004` (default) - 768 dimensions

**Pricing:**
- Free tier available
- Check Google AI Studio for current pricing

**Usage:**
```python
generator = EmbeddingGenerator(
    provider="gemini",
    model_name="text-embedding-004"
)
```

## API Reference

### `EmbeddingGenerator`

#### `__init__(provider="openai", model_name=None, api_key=None)`

Initialize the embedding generator.

**Parameters:**
- `provider` (str): "openai" or "gemini"
- `model_name` (str, optional): Specific model (uses defaults if None)
- `api_key` (str, optional): API key (reads from env if None)

**Raises:**
- `ValueError`: If provider unsupported or API key missing

**Example:**
```python
generator = EmbeddingGenerator(provider="openai")
```

---

#### `generate_embedding(text: str) -> List[float]`

Generate embedding for a single text.

**Parameters:**
- `text` (str): Input text to embed

**Returns:**
- `List[float]`: Vector embedding

**Example:**
```python
embedding = generator.generate_embedding("Hello world")
```

---

#### `generate_embeddings_batch(texts: List[str], show_progress=True) -> List[List[float]]`

Generate embeddings for multiple texts.

**Parameters:**
- `texts` (List[str]): List of texts to embed
- `show_progress` (bool): Whether to log progress

**Returns:**
- `List[List[float]]`: List of embeddings

**Example:**
```python
embeddings = generator.generate_embeddings_batch(chunks)
```

---

#### `get_embedding_dimension() -> int`

Get the dimension of embeddings for the current model.

**Returns:**
- `int`: Number of dimensions

**Example:**
```python
dim = generator.get_embedding_dimension()  # 1536 for OpenAI
```

---

#### `get_provider_info() -> Dict[str, Any]`

Get information about the current provider.

**Returns:**
- `Dict`: Provider details

**Example:**
```python
info = generator.get_provider_info()
print(info)
# {'provider': 'openai', 'model': 'text-embedding-3-small', 
#  'dimensions': 1536, 'api_key_set': True}
```

---

### Convenience Function

#### `generate_embeddings(texts, provider="openai", model_name=None, api_key=None)`

Quick function to generate embeddings without instantiating the class.

**Example:**
```python
from vector_store.embeddings import generate_embeddings

embeddings = generate_embeddings(["Text 1", "Text 2"])
```

## Testing

Run the test suite:

```bash
# Make sure API key is set first
$env:OPENAI_API_KEY="your-key"

# Run tests
python test_embeddings.py
```

**Tests include:**
- Provider initialization
- Single embedding generation
- Batch embedding generation
- Semantic similarity demo
- Convenience function

## Error Handling

### Missing API Key

```python
# ✗ Error
generator = EmbeddingGenerator(provider="openai")
# ValueError: OpenAI API key not found. Set OPENAI_API_KEY environment variable.
```

**Solution:**
```bash
$env:OPENAI_API_KEY="your-key-here"
```

### Unsupported Provider

```python
# ✗ Error
generator = EmbeddingGenerator(provider="unknown")
# ValueError: Unsupported provider: unknown. Supported providers: openai, gemini
```

### Empty Text

```python
embedding = generator.generate_embedding("")
# Returns: []
# Logs warning: "Empty text provided for embedding"
```

## Best Practices

### 1. **Batch Processing**

```python
# ✓ Good - Batch processing
embeddings = generator.generate_embeddings_batch(chunks)

# ✗ Avoid - Individual calls in loop
embeddings = [generator.generate_embedding(c) for c in chunks]
```

### 2. **Choose Right Provider**

| Provider | Pros | Cons |
|----------|------|------|
| **OpenAI** | High quality, 1536 dims | Paid service |
| **Gemini** | Free tier, good quality | 768 dims |

### 3. **Handle Failures**

```python
try:
    embeddings = generator.generate_embeddings_batch(chunks)
except Exception as e:
    logger.error(f"Embedding failed: {e}")
    # Handle error appropriately
```

### 4. **Monitor Costs**

```python
# Estimate cost for OpenAI
num_tokens = sum(len(chunk.split()) for chunk in chunks) * 1.3
cost = (num_tokens / 1_000_000) * 0.02  # $0.02 per 1M tokens
print(f"Estimated cost: ${cost:.4f}")
```

## Performance

**Typical speeds:**
- OpenAI: ~100-200 chunks/minute
- Gemini: ~50-100 chunks/minute

**For 225 chunks (biology PDF):**
- OpenAI: ~1-2 minutes
- Gemini: ~2-4 minutes

## Switching Providers

Easy to switch between providers:

```python
# Development: Use Gemini (free tier)
generator = EmbeddingGenerator(provider="gemini")

# Production: Use OpenAI (higher quality)
generator = EmbeddingGenerator(provider="openai")
```

Or use configuration:

```python
from config import Config

generator = EmbeddingGenerator(
    provider=Config.EMBEDDING_PROVIDER,
    model_name=Config.EMBEDDING_MODEL
)
```

## Next Steps

After generating embeddings:

1. **Store in vector database** (`vector_store/database.py`)
2. **Implement similarity search**
3. **Build RAG retrieval pipeline**
4. **Integrate with teaching engine**

---

**Status**: ✅ Fully implemented and tested  
**Dependencies**: `openai` or `google-generativeai`  
**API Keys Required**: Yes (from environment variables)
