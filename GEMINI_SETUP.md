# Quick Setup Guide - Using Gemini API

## ✅ Gemini is Now the Default!

The project is now configured to use **Gemini** as the default embedding provider.

## Step 1: Get Your Gemini API Key

1. Go to: **https://makersuite.google.com/app/apikey**
2. Sign in with your Google account
3. Click **"Create API Key"**
4. Copy your API key

## Step 2: Set the Environment Variable

### Windows PowerShell (Recommended)

Open PowerShell and run:

```powershell
$env:GEMINI_API_KEY="your-gemini-api-key-here"
```

**Example:**
```powershell
$env:GEMINI_API_KEY="AIzaSyABC123def456GHI789jkl012MNO345pqr"
```

### Verify it's set:

```powershell
echo $env:GEMINI_API_KEY
```

## Step 3: Test It!

Run the embeddings test:

```powershell
python test_embeddings.py
```

Or create a simple test:

```python
from vector_store.embeddings import EmbeddingGenerator

# This will automatically use Gemini (from config)
generator = EmbeddingGenerator(provider="gemini")

# Test with a simple text
embedding = generator.generate_embedding("Hello world")

print(f"✓ Gemini working! Embedding dimension: {len(embedding)}")
# Should print: ✓ Gemini working! Embedding dimension: 768
```

## Alternative: Use .env File (Permanent)

If you want the API key to persist across sessions:

1. **Create a file named `.env`** in `d:\mod\ai_tutor\`

2. **Add this line:**
   ```
   GEMINI_API_KEY=your-gemini-api-key-here
   ```

3. **Install python-dotenv** (if not already installed):
   ```powershell
   pip install python-dotenv
   ```

4. **Load it in your code:**
   ```python
   from dotenv import load_dotenv
   load_dotenv()  # This loads .env file
   
   # Now your code can access the API key
   from vector_store.embeddings import EmbeddingGenerator
   generator = EmbeddingGenerator(provider="gemini")
   ```

## Using Gemini in Your Code

### Option 1: Use Config (Automatic)

```python
from config import Config
from vector_store.embeddings import EmbeddingGenerator

# Uses Gemini automatically (from config)
generator = EmbeddingGenerator(
    provider=Config.EMBEDDING_PROVIDER,  # "gemini"
    model_name=Config.EMBEDDING_MODEL    # "text-embedding-004"
)
```

### Option 2: Specify Directly

```python
from vector_store.embeddings import EmbeddingGenerator

# Explicitly use Gemini
generator = EmbeddingGenerator(provider="gemini")
```

### Option 3: Convenience Function

```python
from vector_store.embeddings import generate_embeddings

# Quick one-liner
embeddings = generate_embeddings(chunks, provider="gemini")
```

## Complete Pipeline with Gemini

```python
from ingestion.document_loader import DocumentLoader
from ingestion.text_processor import chunk_text
from vector_store.embeddings import EmbeddingGenerator

# Load PDF
loader = DocumentLoader()
text = loader.load_pdf("sample.pdf")

# Chunk text
chunks = chunk_text(text, chunk_size=600, chunk_overlap=60)

# Generate embeddings with Gemini
generator = EmbeddingGenerator(provider="gemini")
embeddings = generator.generate_embeddings_batch(chunks)

print(f"✓ Generated {len(embeddings)} embeddings using Gemini!")
print(f"✓ Dimension: {len(embeddings[0])} (Gemini uses 768 dimensions)")
```

## Gemini vs OpenAI

| Feature | Gemini | OpenAI |
|---------|--------|--------|
| **Dimensions** | 768 | 1536 |
| **Free Tier** | ✅ Yes | ❌ No |
| **Quality** | Excellent | Excellent |
| **Speed** | Good | Good |
| **Cost** | Free (with limits) | $0.02/1M tokens |

## Troubleshooting

### Error: "Gemini API key not found"

**Solution:**
```powershell
# Set the environment variable
$env:GEMINI_API_KEY="your-key-here"

# Verify
echo $env:GEMINI_API_KEY
```

### Error: "google-generativeai not installed"

**Solution:**
```powershell
pip install google-generativeai
```

### Want to switch back to OpenAI?

Just change the provider:

```python
generator = EmbeddingGenerator(provider="openai")
```

Or update `config.py`:
```python
EMBEDDING_PROVIDER = "openai"
```

## Next Steps

1. ✅ Set your Gemini API key
2. ✅ Test with `python test_embeddings.py`
3. ✅ Run the complete pipeline on your biology PDF
4. 🔜 Implement vector database storage

---

**Current Status**: Gemini is configured as default ✅  
**API Key Location**: Environment variable `GEMINI_API_KEY`  
**Ready to use**: Yes!
