"""
Test Script for Vector Database (ChromaDB)

This script demonstrates storing and retrieving document chunks using ChromaDB.

Usage:
    python test_vector_database.py
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


def test_database_initialization():
    """Test initializing the vector database."""
    print("=" * 70)
    print("TEST 1: Database Initialization")
    print("=" * 70)
    print()
    
    try:
        from vector_store.database import VectorDatabase
        
        # Initialize database
        db = VectorDatabase(
            collection_name="test_collection",
            persist_directory="./data/test_chroma_db"
        )
        
        print("✓ Database initialized successfully")
        print(f"  Collection: {db.collection_name}")
        print(f"  Persist directory: {db.persist_directory}")
        print(f"  Document count: {db.get_document_count()}")
        print()
        
        return db
        
    except ImportError as e:
        print(f"✗ Failed: {e}")
        print()
        print("Install ChromaDB:")
        print("  pip install chromadb")
        print()
        return None
    except Exception as e:
        print(f"✗ Failed: {e}")
        print()
        return None


def test_add_documents(db):
    """Test adding documents to the database."""
    if not db:
        print("Skipping add documents test (no database)")
        return None
    
    print("=" * 70)
    print("TEST 2: Adding Documents")
    print("=" * 70)
    print()
    
    # Sample biology chunks
    chunks = [
        "Photosynthesis is the process by which plants convert light energy into chemical energy.",
        "Cells are the basic unit of life and contain DNA.",
        "DNA stores genetic information in the form of genes.",
        "Mitochondria are the powerhouse of the cell.",
        "Evolution is the change in species over time through natural selection.",
    ]
    
    # Create simple embeddings (normally these would come from EmbeddingGenerator)
    # For testing, we'll create dummy embeddings
    import random
    random.seed(42)  # For reproducibility
    
    embeddings = []
    for _ in chunks:
        # Create a random 768-dimensional vector (Gemini embedding size)
        embedding = [random.random() for _ in range(768)]
        embeddings.append(embedding)
    
    print(f"Adding {len(chunks)} documents...")
    print()
    
    try:
        count = db.add_documents(chunks, embeddings)
        
        print(f"✓ Successfully added {count} documents")
        print(f"  Total documents in database: {db.get_document_count()}")
        print()
        
        # Show sample chunks
        print("Sample chunks added:")
        for i, chunk in enumerate(chunks[:3], 1):
            print(f"  {i}. {chunk[:60]}...")
        print()
        
        return embeddings
        
    except Exception as e:
        print(f"✗ Failed to add documents: {e}")
        print()
        return None


def test_similarity_search(db, embeddings):
    """Test similarity search."""
    if not db or not embeddings:
        print("Skipping similarity search test")
        return
    
    print("=" * 70)
    print("TEST 3: Similarity Search")
    print("=" * 70)
    print()
    
    # Use the first embedding as a query (should match itself)
    query_embedding = embeddings[0]
    
    print("Query: Using first document's embedding")
    print("Expected: Should find the first document as most similar")
    print()
    
    try:
        results = db.similarity_search(query_embedding, top_k=3)
        
        print(f"✓ Found {len(results)} results")
        print()
        
        print("Top 3 results:")
        for i, (text, score, metadata) in enumerate(results, 1):
            print(f"\n{i}. Similarity: {score:.4f}")
            print(f"   Text: {text[:70]}...")
            print(f"   Metadata: {metadata}")
        
        print()
        
    except Exception as e:
        print(f"✗ Similarity search failed: {e}")
        print()


def test_collection_info(db):
    """Test getting collection information."""
    if not db:
        print("Skipping collection info test")
        return
    
    print("=" * 70)
    print("TEST 4: Collection Information")
    print("=" * 70)
    print()
    
    try:
        info = db.get_collection_info()
        
        print("Collection Details:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        print()
        
    except Exception as e:
        print(f"✗ Failed to get collection info: {e}")
        print()


def test_persistence(db):
    """Test that data persists across sessions."""
    if not db:
        print("Skipping persistence test")
        return
    
    print("=" * 70)
    print("TEST 5: Persistence")
    print("=" * 70)
    print()
    
    try:
        # Get current count
        count_before = db.get_document_count()
        print(f"Documents before: {count_before}")
        
        # Create a new database instance (simulates restart)
        from vector_store.database import VectorDatabase
        
        db2 = VectorDatabase(
            collection_name=db.collection_name,
            persist_directory=db.persist_directory
        )
        
        count_after = db2.get_document_count()
        print(f"Documents after reload: {count_after}")
        print()
        
        if count_before == count_after:
            print("✓ Data persisted successfully!")
        else:
            print("✗ Data did not persist correctly")
        
        print()
        
    except Exception as e:
        print(f"✗ Persistence test failed: {e}")
        print()


def test_cleanup(db):
    """Clean up test data."""
    if not db:
        return
    
    print("=" * 70)
    print("CLEANUP: Clearing Test Data")
    print("=" * 70)
    print()
    
    try:
        success = db.clear_collection()
        
        if success:
            print("✓ Test collection cleared")
            print(f"  Documents remaining: {db.get_document_count()}")
        else:
            print("✗ Failed to clear collection")
        
        print()
        
    except Exception as e:
        print(f"✗ Cleanup failed: {e}")
        print()


def main():
    """Run all tests."""
    print("\n")
    print("=" * 70)
    print("VECTOR DATABASE TEST SUITE (ChromaDB)")
    print("=" * 70)
    print("\n")
    
    try:
        # Test 1: Initialize
        db = test_database_initialization()
        
        if db:
            # Test 2: Add documents
            embeddings = test_add_documents(db)
            
            # Test 3: Similarity search
            test_similarity_search(db, embeddings)
            
            # Test 4: Collection info
            test_collection_info(db)
            
            # Test 5: Persistence
            test_persistence(db)
            
            # Cleanup
            test_cleanup(db)
        
        print("=" * 70)
        print("✓ TEST SUITE COMPLETED")
        print("=" * 70)
        print()
        
    except Exception as e:
        print("=" * 70)
        print(f"✗ TEST SUITE FAILED: {e}")
        print("=" * 70)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
