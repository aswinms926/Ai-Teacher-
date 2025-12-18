"""
Test Script for Text Processor

This script demonstrates text cleaning and chunking functionality.

Usage:
    python test_text_processor.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from ingestion.text_processor import TextProcessor


def test_text_cleaning():
    """Test the text cleaning functionality."""
    print("=" * 70)
    print("TEST 1: Text Cleaning")
    print("=" * 70)
    print()
    
    processor = TextProcessor()
    
    # Test cases for cleaning
    test_cases = [
        ("Hello....    world!!!", "Excessive dots and spaces"),
        ("This  is    a   test....", "Multiple spaces"),
        ("No space.After punctuation", "Missing space after period"),
        ('Smart "quotes" and \'apostrophes\'', "Smart quotes"),
        ("Multiple\n\n\n\nnewlines", "Excessive newlines"),
        ("Tabs\t\there", "Tab characters"),
    ]
    
    for dirty_text, description in test_cases:
        clean_text = processor.clean_text(dirty_text)
        print(f"Test: {description}")
        print(f"  Input:  '{dirty_text}'")
        print(f"  Output: '{clean_text}'")
        print()


def test_text_chunking():
    """Test the text chunking functionality."""
    print("=" * 70)
    print("TEST 2: Text Chunking")
    print("=" * 70)
    print()
    
    # Sample educational text
    sample_text = """
    Introduction to Machine Learning
    
    Machine learning is a subset of artificial intelligence that focuses on 
    building systems that can learn from data. These systems improve their 
    performance on a specific task over time without being explicitly programmed.
    
    Types of Machine Learning
    
    There are three main types of machine learning: supervised learning, 
    unsupervised learning, and reinforcement learning. Each type has its own 
    use cases and applications.
    
    Supervised Learning
    
    In supervised learning, the algorithm learns from labeled training data. 
    The goal is to learn a mapping from inputs to outputs. Common examples 
    include classification and regression tasks.
    
    Unsupervised Learning
    
    Unsupervised learning works with unlabeled data. The algorithm tries to 
    find patterns and structure in the data without explicit guidance. 
    Clustering and dimensionality reduction are common unsupervised learning tasks.
    
    Reinforcement Learning
    
    Reinforcement learning involves an agent learning to make decisions by 
    interacting with an environment. The agent receives rewards or penalties 
    based on its actions and learns to maximize cumulative reward over time.
    
    Applications of Machine Learning
    
    Machine learning has numerous applications across various domains. In 
    healthcare, it's used for disease diagnosis and drug discovery. In finance, 
    it powers fraud detection and algorithmic trading. In technology, it enables 
    recommendation systems, natural language processing, and computer vision.
    
    Conclusion
    
    Machine learning continues to evolve and transform industries. As data 
    becomes more abundant and computing power increases, we can expect even 
    more innovative applications of machine learning in the future.
    """
    
    # Test with different chunk sizes
    chunk_sizes = [300, 500, 800]
    
    for chunk_size in chunk_sizes:
        print(f"\n--- Chunk Size: {chunk_size} characters ---\n")
        
        processor = TextProcessor(chunk_size=chunk_size, chunk_overlap=50)
        chunks = processor.chunk_text(sample_text)
        
        print(f"Total chunks created: {len(chunks)}")
        print(f"Average chunk size: {sum(len(c) for c in chunks) // len(chunks)} chars")
        print()
        
        # Show each chunk
        for i, chunk in enumerate(chunks, 1):
            print(f"Chunk {i} ({len(chunk)} chars):")
            print("-" * 70)
            # Show first 200 characters of each chunk
            preview = chunk[:200] + ("..." if len(chunk) > 200 else "")
            print(preview)
            print()


def test_metadata_extraction():
    """Test metadata extraction."""
    print("=" * 70)
    print("TEST 3: Metadata Extraction")
    print("=" * 70)
    print()
    
    sample_text = """
    Chapter 1: Introduction
    
    This is the introduction to our comprehensive guide on artificial intelligence.
    We will cover various topics including machine learning, deep learning, and
    natural language processing.
    
    What is AI?
    
    Artificial Intelligence (AI) refers to the simulation of human intelligence
    in machines. These machines are programmed to think and learn like humans.
    """
    
    processor = TextProcessor()
    metadata = processor.extract_metadata(sample_text)
    
    print("Extracted Metadata:")
    print("-" * 70)
    for key, value in metadata.items():
        print(f"{key:20s}: {value}")
    print()


def test_edge_cases():
    """Test edge cases and error handling."""
    print("=" * 70)
    print("TEST 4: Edge Cases")
    print("=" * 70)
    print()
    
    processor = TextProcessor(chunk_size=500, chunk_overlap=50)
    
    # Test 1: Empty text
    print("Test: Empty text")
    chunks = processor.chunk_text("")
    print(f"  Result: {len(chunks)} chunks (expected: 0)")
    print()
    
    # Test 2: Very short text
    print("Test: Very short text")
    chunks = processor.chunk_text("Hello world!")
    print(f"  Result: {len(chunks)} chunks")
    print(f"  Content: '{chunks[0] if chunks else 'N/A'}'")
    print()
    
    # Test 3: Single very long sentence
    print("Test: Single very long sentence (no natural breaks)")
    long_sentence = "This is a very long sentence " * 50
    chunks = processor.chunk_text(long_sentence)
    print(f"  Result: {len(chunks)} chunks")
    print(f"  Avg size: {sum(len(c) for c in chunks) // len(chunks) if chunks else 0} chars")
    print()
    
    # Test 4: Text with only newlines
    print("Test: Text with excessive newlines")
    newline_text = "\n\n\n\nSome text\n\n\n\nMore text\n\n\n\n"
    chunks = processor.chunk_text(newline_text)
    print(f"  Result: {len(chunks)} chunks")
    print()


def test_overlap_functionality():
    """Test chunk overlap functionality."""
    print("=" * 70)
    print("TEST 5: Chunk Overlap")
    print("=" * 70)
    print()
    
    sample_text = """
    First paragraph with some content that will be split into chunks.
    This helps us see how the overlap works between consecutive chunks.
    
    Second paragraph continues the discussion and provides more context.
    The overlap ensures that context is preserved across chunk boundaries.
    
    Third paragraph adds even more information to demonstrate the chunking.
    We want to make sure that important context isn't lost at boundaries.
    """
    
    processor = TextProcessor(chunk_size=150, chunk_overlap=30)
    chunks = processor.chunk_text(sample_text)
    
    print(f"Created {len(chunks)} chunks with 30-character overlap")
    print()
    
    # Show overlap between consecutive chunks
    for i in range(len(chunks) - 1):
        chunk1_end = chunks[i][-50:]  # Last 50 chars of chunk
        chunk2_start = chunks[i + 1][:50]  # First 50 chars of next chunk
        
        print(f"Chunk {i + 1} ending: ...{chunk1_end}")
        print(f"Chunk {i + 2} starting: {chunk2_start}...")
        print()


def main():
    """Run all tests."""
    print("\n")
    print("=" * 70)
    print("TEXT PROCESSOR TEST SUITE")
    print("=" * 70)
    print("\n")
    
    try:
        test_text_cleaning()
        test_text_chunking()
        test_metadata_extraction()
        test_edge_cases()
        test_overlap_functionality()
        
        print("=" * 70)
        print("✓ ALL TESTS COMPLETED SUCCESSFULLY")
        print("=" * 70)
        
    except Exception as e:
        print("=" * 70)
        print(f"✗ TEST FAILED: {e}")
        print("=" * 70)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
