# AI Tutor Final System Overview

## 🏗️ System Architecture

The AI Tutor is built as a **RAG (Retrieval-Augmented Generation)** application with offline-first capabilities.

### Layers

1.  **Frontend (UI Layer)**:
    -   **Technology**: Streamlit (`ui/web_app.py`)
    -   **Function**: Handles user interaction (PDF upload, teaching requests) and displays lectures.
    -   **State**: Manages session state (processed PDFs, DB connections).

2.  **Application Logic (Teaching Layer)**:
    -   **Component**: `RAGTeachingEngine` (`teaching/rag_engine.py`)
    -   **Function**: Orchestrates the teaching process.
    -   **Logic**:
        -   Checks Cache → `lectures/<topic>.txt`
        -   Retrieves Context → `VectorDatabase`
        -   Generates content → `OFFLINE` formatter or `LLM` (Gemini).

3.  **Data Layer (Storage & Retrieval)**:
    -   **Component**: `VectorDatabase` (`vector_store/database.py`)
    -   **Technology**: ChromaDB (Persistent local storage).
    -   **Component**: `EmbeddingGenerator` (`vector_store/embeddings.py`)
    -   **Technology**: Gemini Embeddings (`text-embedding-004`).

4.  **Ingestion Layer**:
    -   **Components**: `DocumentLoader`, `chunk_text`.
    -   **Function**: Converts raw PDFs into indexable vector chunks.

---

## 🔄 RAG Pipeline & Caching

### 1. Ingestion (One-Time)
`PDF -> Text -> Chunks -> Embeddings -> ChromaDB`
-   **Cost**: 1 API call per chunk (embeddings).
-   **Result**: Persistent vector index on disk.

### 2. Teaching Flow (Per Request)

**Step 1: Cache Check**
-   Checks `lectures/<sanitized_topic>.txt`.
-   **Hit**: Returns content instantly (0 Latency, 0 Cost).
-   **Miss**: Proceeds to Step 2.

**Step 2: Retrieval**
-   Converts topic to embedding (1 API call).
-   Queries ChromaDB for top-k relevant chunks.

**Step 3: Generation (Mode Dependent)**

*   **OFFLINE MODE (Default)**:
    -   Uses a deterministic algorithm to format retrieved chunks.
    -   Structure: Intro (Chunk 1) -> Body (Chunks 1-3) -> Summary (Last Chunk).
    -   **Pros**: Works without internet/API, zero cost, impossible to hallucinate.
    -   **Cons**: Less conversational flow than LLM.

*   **LLM MODE (Optional)**:
    -   Sends prompt + context to Gemini (`gemini-pro-latest`).
    -   **Pros**: Natural language, coherent synthesis.
    -   **Cons**: Requires API key, small cost.
    -   **Fallback**: If API fails, automatically switches to OFFLINE mode.

**Step 4: Caching**
-   The generated lecture is saved to disk for future Step 1 hits.

---

## 💻 Running the System

### 1. Requirements
-   Python 3.10+
-   `pip install -r requirements.txt` (including streamlit, chromadb, google-generativeai)
-   Gemini API Key (for Ingestion and LLM mode)

### 2. Launch
```bash
streamlit run app.py
```
*Or simply run `python app.py` which wraps the command.*

### 3. Usage
1.  **Upload Tab**: Upload your textbook PDF. Wait for "Processing Complete".
2.  **Teaching Tab**:
    -   Select "OFFLINE" or "LLM" mode in the sidebar.
    -   Enter a topic (e.g., "Photosynthesis").
    -   Get your lecture!

---

## 🛡️ Robustness & Safety

-   **API Independence**: The system (UI + Offline Mode) functions completely without the Gemini text generation API. Embeddings are required only for new topic queries or new PDF ingestion.
-   **Graceful Degradation**: LLM failures seamlessly fall back to offline generation.
-   **Persistence**: Both vector data and cached lectures survive restarts.
-   **Hallucination Control**:
    -   Offline mode uses *exact* text extracts.
    -   LLM mode is strictly prompted to use *only* provided context.
