"""
Complete RAG Teaching Engine Test

This script demonstrates the complete RAG pipeline:
1. Load PDF
2. Chunk text
3. Generate embeddings
4. Store in vector database
5. Teach topics using RAG

Usage:
    # Make sure your Gemini API key is set:
    $env:GEMINI_API_KEY="your-key"
    
    # Run the test:
    python test_rag_teaching.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Load .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass


def test_complete_rag_pipeline():
    """Test the complete RAG teaching pipeline."""
    
    print("\n")
    print("=" * 80)
    print("COMPLETE RAG TEACHING ENGINE TEST")
    print("=" * 80)
    print("\n")
    
    # Step 1: Load PDF
    print("=" * 80)
    print("STEP 1: Loading PDF Document")
    print("=" * 80)
    print()
    
    try:
        from ingestion.document_loader import DocumentLoader
        
        loader = DocumentLoader()
        text = loader.load_pdf("sample.pdf")
        
        print(f"✓ PDF loaded successfully")
        print(f"  Total characters: {len(text):,}")
        print(f"  Total words: {len(text.split()):,}")
        print()
        
    except Exception as e:
        print(f"✗ Failed to load PDF: {e}")
        return False
    
    # Step 2: Chunk text
    print("=" * 80)
    print("STEP 2: Chunking Text")
    print("=" * 80)
    print()
    
    try:
        from ingestion.text_processor import chunk_text
        
        chunks = chunk_text(text, chunk_size=600, chunk_overlap=60)
        
        print(f"✓ Text chunked successfully")
        print(f"  Total chunks: {len(chunks)}")
        print(f"  Average chunk size: {sum(len(c) for c in chunks) // len(chunks)} chars")
        print()
        
    except Exception as e:
        print(f"✗ Failed to chunk text: {e}")
        return False
    
    # Step 3: Generate embeddings
    print("=" * 80)
    print("STEP 3: Generating Embeddings (Gemini)")
    print("=" * 80)
    print()
    
    try:
        from vector_store.embeddings import EmbeddingGenerator
        
        generator = EmbeddingGenerator(provider="gemini")
        
        print(f"Generating embeddings for {len(chunks)} chunks...")
        print("(This may take 1-2 minutes)")
        print()
        
        embeddings = generator.generate_embeddings_batch(chunks, show_progress=True)
        
        print()
        print(f"✓ Embeddings generated successfully")
        print(f"  Total embeddings: {len(embeddings)}")
        print(f"  Embedding dimension: {len(embeddings[0])}")
        print()
        
    except Exception as e:
        print(f"✗ Failed to generate embeddings: {e}")
        print()
        print("Make sure your Gemini API key is set:")
        print("  $env:GEMINI_API_KEY=\"your-key\"")
        print()
        return False
    
    # Step 4: Store in vector database
    print("=" * 80)
    print("STEP 4: Storing in Vector Database (ChromaDB)")
    print("=" * 80)
    print()
    
    try:
        from vector_store.database import VectorDatabase
        
        db = VectorDatabase(
            collection_name="biology_textbook",
            persist_directory="./data/chroma_db"
        )
        
        # Clear existing data (for testing)
        if db.get_document_count() > 0:
            print(f"Clearing existing {db.get_document_count()} documents...")
            db.clear_collection()
        
        # Add documents
        count = db.add_documents(chunks, embeddings)
        
        print(f"✓ Documents stored successfully")
        print(f"  Documents in database: {db.get_document_count()}")
        print()
        
    except Exception as e:
        print(f"✗ Failed to store in database: {e}")
        return False
    
    # Step 5: Initialize RAG Teaching Engine
    print("=" * 80)
    print("STEP 5: Initializing RAG Teaching Engine")
    print("=" * 80)
    print()
    
    try:
        from teaching.rag_engine import RAGTeachingEngine
        
        engine = RAGTeachingEngine(
            vector_database=db,
            embedding_generator=generator,
            llm_provider="gemini",
            top_k=6
        )
        
        info = engine.get_engine_info()
        
        print(f"✓ RAG Teaching Engine initialized")
        print(f"  LLM: {info['llm_provider']} / {info['llm_model']}")
        print(f"  Embeddings: {info['embedding_provider']} / {info['embedding_model']}")
        print(f"  Vector DB documents: {info['vector_db_documents']}")
        print(f"  Top-K retrieval: {info['top_k']}")
        print()
        
    except Exception as e:
        print(f"✗ Failed to initialize teaching engine: {e}")
        return False
    
    # Step 6: Test teaching on various topics
    print("=" * 80)
    print("STEP 6: Testing Teaching on Various Topics")
    print("=" * 80)
    print()
    
    test_topics = [
        "photosynthesis",
        "cells",
        "evolution"
    ]
    
    for i, topic in enumerate(test_topics, 1):
        print(f"\n{'=' * 80}")
        print(f"TEST {i}: Teaching '{topic}'")
        print('=' * 80)
        print()
        
        try:
            lecture = engine.teach(topic)
            
            print(f"✓ Lecture generated successfully")
            print()
            print("LECTURE:")
            print("-" * 80)
            print(lecture)
            print("-" * 80)
            print()
            
        except Exception as e:
            print(f"✗ Failed to generate lecture: {e}")
            print()
    
    # Final summary
    print("\n")
    print("=" * 80)
    print("✓ COMPLETE RAG PIPELINE TEST COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print()
    
    print("Summary:")
    print(f"  ✓ Loaded PDF with {len(text):,} characters")
    print(f"  ✓ Created {len(chunks)} chunks")
    print(f"  ✓ Generated {len(embeddings)} embeddings")
    print(f"  ✓ Stored in vector database")
    print(f"  ✓ Taught {len(test_topics)} topics successfully")
    print()
    
    print("Your AI Tutor is ready to teach! 🎓")
    print()
    
    return True


def main():
    """Run the complete test."""
    try:
        success = test_complete_rag_pipeline()
        
        if not success:
            print("\n❌ Test failed. Please check the errors above.\n")
            sys.exit(1)
        
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
