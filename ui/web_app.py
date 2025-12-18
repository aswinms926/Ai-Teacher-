"""
AI Tutor Web Interface using Streamlit.

This module provides a simple, teacher-friendly UI for:
1. Uploading and processing PDF textbooks.
2. Configuring teaching mode (Offline vs LLM).
3. Requesting lectures on specific topics.
"""

import streamlit as st
import os
import sys
from pathlib import Path
import tempfile

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

from vector_store.database import VectorDatabase
from vector_store.embeddings import EmbeddingGenerator
from teaching.rag_engine import RAGTeachingEngine
from ingestion.document_loader import DocumentLoader
from ingestion.text_processor import chunk_text

# Page Config
st.set_page_config(
    page_title="AI Tutor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Session State ---
if 'processed_pdf' not in st.session_state:
    st.session_state.processed_pdf = False
if 'db_ready' not in st.session_state:
    st.session_state.db_ready = False

# --- Sidebar Configuration ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/reading.png", width=80)
    st.title("Settings")
    
    st.header("Teaching Mode")
    teaching_mode = st.radio(
        "Select Mode:",
        ["OFFLINE", "LLM"],
        help="OFFLINE: Fast, free, no API. LLM: High quality, uses Gemini."
    )
    
    st.divider()
    
    # API Key check
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if api_key:
        st.success("✅ Gemini API Key found")
    else:
        st.warning("⚠️ No Gemini API Key found")
        if teaching_mode == "LLM":
            st.error("LLM mode requires an API key!")
    
    st.divider()
    st.info(
        "**System Status**\n"
        f"• Mode: {teaching_mode}\n"
        f"• DB Ready: {'Yes' if st.session_state.db_ready else 'No'}"
    )

# --- Main Area ---
st.title("🎓 AI Tutor System")
st.markdown("Upload a textbook PDF and get instant lectures on any topic.")

# --- Tab 1: Upload & Process ---
tab1, tab2 = st.tabs(["📚 Textbook Upload", "🧑‍🏫 Teaching Room"])

with tab1:
    st.header("1. Upload Textbook")
    
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        if st.button("Process PDF", type="primary"):
            with st.status("Processing Textbook...", expanded=True) as status:
                try:
                    # 1. Save temp file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    st.write("✓ PDF saved locally")
                    
                    # 2. Load PDF
                    st.write("📖 Reading PDF content...")
                    loader = DocumentLoader()
                    text = loader.load_pdf(tmp_path)
                    st.write(f"✓ Loaded {len(text):,} characters")
                    
                    # 3. Chunk
                    st.write("✂️ Chunking text...")
                    chunks = chunk_text(text, chunk_size=600, chunk_overlap=60)
                    st.write(f"✓ Created {len(chunks)} text chunks")
                    
                    # 4. Embeddings
                    st.write("🧠 Generating embeddings (Gemini)...")
                    generator = EmbeddingGenerator(provider="gemini")
                    embeddings = generator.generate_embeddings_batch(chunks)
                    st.write(f"✓ Generated {len(embeddings)} embeddings")
                    
                    # 5. Filter successful embeddings
                    valid_chunks = []
                    valid_embeddings = []
                    for i, emb in enumerate(embeddings):
                        if emb and len(emb) > 0:
                            valid_chunks.append(chunks[i])
                            valid_embeddings.append(emb)
                    
                    if len(valid_chunks) < len(chunks):
                        st.warning(f"⚠️ Failed to generate embeddings for {len(chunks) - len(valid_chunks)} chunks (likely due to rate limits). Proceeding with {len(valid_chunks)} valid chunks.")
                    
                    if not valid_chunks:
                        st.error("❌ Failed to generate any embeddings. Please check your API key and quotas.")
                        status.update(label="❌ Embedding Failed", state="error")
                        st.stop()

                    # 6. Store
                    st.write("💾 Storing in Vector Database...")
                    # Re-initialize DB to ensure clean state or append? 
                    # For demo, we might want to clear old data or use a specific collection
                    db = VectorDatabase(collection_name="ai_tutor_demo", persist_directory="./data/chroma_db_demo")
                    db.clear_collection() # Fresh start for demo
                    db.add_documents(valid_chunks, valid_embeddings)
                    st.write(f"✓ Saved {len(valid_chunks)} documents to ChromaDB")
                    
                    # Cleanup
                    os.unlink(tmp_path)
                    
                    st.session_state.processed_pdf = True
                    st.session_state.db_ready = True
                    status.update(label="✅ Processing Complete!", state="complete", expanded=False)
                    st.balloons()
                    
                except Exception as e:
                    st.error(f"Processing failed: {e}")
                    status.update(label="❌ Failed", state="error")

with tab2:
    st.header("2. Teaching Room")
    
    if not st.session_state.db_ready:
        # Check if DB exists on disk from previous run
        if os.path.exists("./data/chroma_db_demo"):
             st.info("Found existing database. Ready to teach!")
             st.session_state.db_ready = True
        else:
            st.warning("Please upload and process a textbook first in the 'Textbook Upload' tab.")
            st.stop()
    
    # Initialize Engine
    try:
        db = VectorDatabase(collection_name="ai_tutor_demo", persist_directory="./data/chroma_db_demo")
        embedder = EmbeddingGenerator(provider="gemini")
        
        # Initialize engine with selected mode
        engine = RAGTeachingEngine(
            vector_database=db,
            embedding_generator=embedder,
            teaching_mode=teaching_mode,
            cache_dir="./lectures"
        )
    except Exception as e:
        st.error(f"Failed to initialize engine: {e}")
        st.stop()
        
    # Input
    col1, col2 = st.columns([3, 1])
    with col1:
        topic = st.text_input("Enter a topic to teach:", placeholder="e.g. Photosynthesis, Cellular Respiration")
    with col2:
        st.write("") # Spacer
        st.write("") # Spacer
        teach_btn = st.button("Start Lecture 🎓", type="primary", use_container_width=True)
    
    if teach_btn and topic:
        with st.spinner(f"Preparing lecture on '{topic}'..."):
            try:
                lecture_content = engine.teach(topic)
                
                # Display output
                st.subheader(f"Lecture: {topic.title()}")
                st.markdown(lecture_content)
                
                # Show source info
                if engine.teaching_mode == "OFFLINE":
                    st.caption("⚡ Offline Mode | 💾 Loaded from Database/Cache")
                else:
                    st.caption(f"🤖 LLM Mode ({engine.llm_model}) | 💾 Saved to Cache")
                    
            except Exception as e:
                st.error(f"An error occurred: {e}")

# Footer
st.markdown("---")
st.markdown("AI Tutor System | v1.0 | Built with Streamlit, Gemini & ChromaDB")
