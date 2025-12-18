# Gemini API Usage in AI Tutor System

## Overview

Your AI Tutor uses the Gemini API in **TWO** places:

---

## 1️⃣ **Embeddings Generation** (Phase 3)

**File**: `vector_store/embeddings.py`

**API Used**: `genai.embed_content()`

**Model**: `text-embedding-004`

**Purpose**: Convert text chunks into vector embeddings (768 dimensions)

**When It's Called**:
- When processing the PDF textbook
- When converting student queries to embeddings for search

**Code Location**:
```python
# Line ~180 in vector_store/embeddings.py
result = genai.embed_content(
    model="models/text-embedding-004",
    content=text,
    task_type="retrieval_document"
)
embedding = result['embedding']
```

**API Calls**:
- **During setup**: 181 calls (one per chunk from biology PDF)
- **During teaching**: 1 call per topic (to embed the query)

**Example**:
```python
from vector_store.embeddings import EmbeddingGenerator

generator = EmbeddingGenerator(provider="gemini")
embedding = generator.generate_embedding("photosynthesis")
# ☝️ This calls Gemini API once
```

---

## 2️⃣ **Text Generation / Teaching** (Phase 4A)

**File**: `teaching/rag_engine.py`

**API Used**: `genai.GenerativeModel().generate_content()`

**Model**: `models/gemini-pro-latest`

**Purpose**: Generate structured lectures from retrieved textbook content

**When It's Called**:
- Every time you call `engine.teach(topic)`

**Code Location**:
```python
# Line ~295 in teaching/rag_engine.py
response = self.llm_client.generate_content(prompt)
lecture = response.text
```

**API Calls**:
- **1 call per topic taught**

**Example**:
```python
from teaching.rag_engine import RAGTeachingEngine

engine = RAGTeachingEngine(db, embedder)
lecture = engine.teach("photosynthesis")
# ☝️ This calls Gemini API once
```

---

## Complete API Call Breakdown

### Initial Setup (One-time):
```
1. Load PDF → No API calls
2. Chunk text → No API calls
3. Generate embeddings → 181 API calls (Gemini Embeddings)
4. Store in database → No API calls
```

### Teaching a Topic:
```
1. Convert topic to embedding → 1 API call (Gemini Embeddings)
2. Search vector database → No API calls (local ChromaDB)
3. Generate lecture → 1 API call (Gemini Text Generation)
```

**Total per topic**: 2 API calls

---

## API Call Summary

| Operation | API | Model | Calls | When |
|-----------|-----|-------|-------|------|
| **Setup: Embed chunks** | Gemini Embeddings | text-embedding-004 | 181 | One-time |
| **Query: Embed topic** | Gemini Embeddings | text-embedding-004 | 1 | Per topic |
| **Generate lecture** | Gemini Text Gen | gemini-pro-latest | 1 | Per topic |

---

## Rate Limits (Free Tier)

**Gemini Free Tier Limits**:
- **Embeddings**: 1,500 requests/day
- **Text Generation**: 60 requests/minute, 1,500 requests/day

**Your Usage**:
- **Setup**: 181 embedding calls (well within limit)
- **Teaching**: 2 calls per topic (also within limit)

**If you hit rate limits**:
- Wait 48 seconds (as shown in error)
- Or upgrade to paid tier for higher limits

---

## Where API Keys Are Used

**Environment Variable**: `GEMINI_API_KEY` or `GOOGLE_API_KEY`

**Set in**: `.env` file or PowerShell

**Used in**:
1. `vector_store/embeddings.py` (line ~127)
2. `teaching/rag_engine.py` (line ~127)

**Code**:
```python
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)
```

---

## What Does NOT Use API

✅ **No API calls for**:
- PDF loading (`ingestion/document_loader.py`)
- Text chunking (`ingestion/text_processor.py`)
- Vector database storage (`vector_store/database.py`)
- Vector database search (`vector_store/database.py`)

These are all **local operations** - no internet required!

---

## Cost Estimate (If Using Paid Tier)

**Gemini Pricing** (as of 2024):
- **Embeddings**: Free (currently)
- **Text Generation**: ~$0.00025 per 1K characters

**Your Biology Textbook**:
- **Setup**: Free (embeddings)
- **Per lecture**: ~$0.001 (assuming 4K character response)

**Very affordable!** 💰

---

## API Flow Diagram

```
┌─────────────────────────────────────────────────────┐
│                   SETUP (One-time)                  │
├─────────────────────────────────────────────────────┤
│                                                     │
│  PDF → Chunks → [Gemini Embeddings API] → Database │
│                      ↑                              │
│                 181 API calls                       │
│                                                     │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│              TEACHING (Per Topic)                   │
├─────────────────────────────────────────────────────┤
│                                                     │
│  Topic → [Gemini Embeddings API] → Query Embedding │
│                ↑                                    │
│           1 API call                                │
│                ↓                                    │
│  Query Embedding → Vector DB Search → Top Chunks   │
│                                                     │
│  Top Chunks → [Gemini Text Gen API] → Lecture      │
│                      ↑                              │
│                 1 API call                          │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## Summary

**You use Gemini API in 2 places**:

1. **`vector_store/embeddings.py`** - Convert text to embeddings
2. **`teaching/rag_engine.py`** - Generate lectures

**Total API calls per teaching session**:
- Setup: 181 calls (one-time)
- Per topic: 2 calls (1 embedding + 1 generation)

**All other operations are local** (no API, no cost, no rate limits)!

---

**Current Status**: ✅ Both APIs working correctly  
**Rate Limits**: Free tier (1,500 requests/day)  
**Cost**: Free tier (or ~$0.001 per lecture on paid tier)
