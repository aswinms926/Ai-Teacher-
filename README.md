# AI Tutor - RAG-based Teaching System

An intelligent tutoring system that uses Retrieval-Augmented Generation (RAG) to provide personalized educational assistance.

## 🏗️ Project Structure

```
ai_tutor/
├── ingestion/              # Document loading and processing
│   ├── __init__.py
│   ├── document_loader.py  # Load PDFs, text files, web pages
│   └── text_processor.py   # Chunk and preprocess text
│
├── vector_store/           # Embedding and vector database
│   ├── __init__.py
│   ├── embeddings.py       # Generate embeddings
│   └── database.py         # Vector database operations
│
├── teaching/               # Core RAG teaching logic
│   ├── __init__.py
│   ├── rag_engine.py       # Main RAG engine
│   └── prompt_templates.py # Teaching prompts
│
├── ui/                     # User interfaces
│   ├── __init__.py
│   ├── cli.py             # Command-line interface
│   └── web_app.py         # Web interface (Streamlit/Gradio)
│
├── config.py              # Configuration settings
├── main.py                # Main entry point
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip or conda

### Installation

1. **Clone or navigate to the project directory**
   ```bash
   cd d:\mod\ai_tutor
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   venv\Scripts\activate  # On Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file in the project root:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

### Usage

**Run the AI Tutor:**
```bash
python main.py
```

## 📋 Development Roadmap

### Phase 1: Data Ingestion
- [ ] Implement document loaders (PDF, text, web)
- [ ] Implement text chunking and preprocessing
- [ ] Test with sample educational content

### Phase 2: Vector Store
- [ ] Set up ChromaDB or alternative
- [ ] Implement embedding generation
- [ ] Test similarity search

### Phase 3: Teaching Engine
- [ ] Integrate RAG engine
- [ ] Design teaching prompts
- [ ] Implement conversation management

### Phase 4: User Interface
- [ ] Build CLI interface
- [ ] Build web interface
- [ ] Add document upload feature

### Phase 5: Enhancement
- [ ] Add conversation history
- [ ] Implement student progress tracking
- [ ] Add multi-modal support (images, diagrams)

## 🛠️ Technology Stack

- **Embeddings**: Sentence Transformers / OpenAI
- **Vector DB**: ChromaDB / Pinecone / FAISS
- **LLM**: OpenAI GPT / Anthropic Claude
- **UI**: Streamlit / Gradio
- **Document Processing**: PyPDF2, BeautifulSoup

## 📝 Notes

- All files currently contain boilerplate code with TODO comments
- Implementation will be done in phases
- Configuration can be customized in `config.py`

## 🤝 Contributing

This is a structured template ready for implementation. Follow the TODO comments in each file to build out the functionality.

---

**Status**: 🏗️ Project structure created - Ready for implementation
