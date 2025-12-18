"""
Test Script for Embeddings Generator

This script demonstrates how to generate embeddings using different providers.

Usage:
    # Set your API key first:
    # For OpenAI:
    #   export OPENAI_API_KEY="your-key-here"  (Linux/Mac)
    #   $env:OPENAI_API_KEY="your-key-here"    (Windows PowerShell)
    
    # For Gemini:
    #   export GEMINI_API_KEY="your-key-here"  (Linux/Mac)
    #   $env:GEMINI_API_KEY="your-key-here"    (Windows PowerShell)
    
    # Then run:
    python test_embeddings.py
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from vector_store.embeddings import EmbeddingGenerator, generate_embeddings


def test_provider_info():
    """Test getting provider information."""
    print("=" * 70)
    print("TEST 1: Provider Information")
    print("=" * 70)
    print()
    
    # Check if API keys are set
    openai_key = os.getenv("OPENAI_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    
    print("API Key Status:")
    print(f"  OpenAI: {'✓ Set' if openai_key else '✗ Not set'}")
    print(f"  Gemini: {'✓ Set' if gemini_key else '✗ Not set'}")
    print()
    
    if not openai_key and not gemini_key:
        print("⚠ WARNING: No API keys found!")
        print()
        print("Please set at least one API key:")
        print("  For OpenAI: $env:OPENAI_API_KEY='your-key'")
        print("  For Gemini: $env:GEMINI_API_KEY='your-key'")
        print()
        return None
    
    # Test with available provider
    provider = "openai" if openai_key else "gemini"
    
    try:
        generator = EmbeddingGenerator(provider=provider)
        info = generator.get_provider_info()
        
        print(f"✓ Successfully initialized {provider.upper()} provider")
        print()
        print("Provider Details:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        print()
        
        return generator
        
    except Exception as e:
        print(f"✗ Failed to initialize: {e}")
        return None


def test_single_embedding(generator):
    """Test generating a single embedding."""
    if not generator:
        print("Skipping single embedding test (no generator)")
        return
    
    print("=" * 70)
    print("TEST 2: Single Embedding Generation")
    print("=" * 70)
    print()
    
    test_text = "Machine learning is a subset of artificial intelligence."
    
    try:
        print(f"Input text: '{test_text}'")
        print()
        
        embedding = generator.generate_embedding(test_text)
        
        print(f"✓ Embedding generated successfully")
        print(f"  Dimensions: {len(embedding)}")
        print(f"  First 5 values: {embedding[:5]}")
        print(f"  Data type: {type(embedding)}")
        print()
        
    except Exception as e:
        print(f"✗ Failed to generate embedding: {e}")
        print()


def test_batch_embeddings(generator):
    """Test generating embeddings for multiple texts."""
    if not generator:
        print("Skipping batch embedding test (no generator)")
        return
    
    print("=" * 70)
    print("TEST 3: Batch Embedding Generation")
    print("=" * 70)
    print()
    
    # Sample texts from biology
    test_texts = [
        "Biology is the study of living organisms.",
        "Cells are the basic unit of life.",
        "DNA contains genetic information.",
        "Photosynthesis converts light energy into chemical energy.",
        "Evolution explains the diversity of life on Earth.",
    ]
    
    print(f"Generating embeddings for {len(test_texts)} texts...")
    print()
    
    try:
        embeddings = generator.generate_embeddings_batch(
            test_texts,
            show_progress=False  # Disable progress for small batch
        )
        
        print(f"✓ Batch embedding completed")
        print(f"  Total embeddings: {len(embeddings)}")
        print(f"  Embedding dimension: {len(embeddings[0]) if embeddings else 0}")
        print()
        
        # Show sample
        print("Sample embeddings (first 3 values of each):")
        for i, (text, emb) in enumerate(zip(test_texts, embeddings), 1):
            preview = text[:50] + "..." if len(text) > 50 else text
            print(f"  {i}. '{preview}'")
            print(f"     → [{emb[0]:.4f}, {emb[1]:.4f}, {emb[2]:.4f}, ...]")
        print()
        
        return embeddings
        
    except Exception as e:
        print(f"✗ Failed to generate batch embeddings: {e}")
        print()
        return None


def test_similarity_demo(embeddings):
    """Demonstrate semantic similarity using embeddings."""
    if not embeddings or len(embeddings) < 2:
        print("Skipping similarity demo (insufficient embeddings)")
        return
    
    print("=" * 70)
    print("TEST 4: Semantic Similarity Demo")
    print("=" * 70)
    print()
    
    import math
    
    def cosine_similarity(vec1, vec2):
        """Calculate cosine similarity between two vectors."""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))
        return dot_product / (magnitude1 * magnitude2)
    
    # Compare first embedding with all others
    base_idx = 0
    base_text = "Biology is the study of living organisms."
    
    print(f"Comparing: '{base_text}'")
    print("With other texts:")
    print()
    
    texts = [
        "Biology is the study of living organisms.",
        "Cells are the basic unit of life.",
        "DNA contains genetic information.",
        "Photosynthesis converts light energy into chemical energy.",
        "Evolution explains the diversity of life on Earth.",
    ]
    
    for i, (text, emb) in enumerate(zip(texts, embeddings)):
        if i == base_idx:
            continue
        
        similarity = cosine_similarity(embeddings[base_idx], emb)
        print(f"  {i + 1}. '{text}'")
        print(f"     Similarity: {similarity:.4f}")
        print()
    
    print("Note: Higher similarity (closer to 1.0) means more semantically similar")
    print()


def test_convenience_function():
    """Test the convenience function."""
    print("=" * 70)
    print("TEST 5: Convenience Function")
    print("=" * 70)
    print()
    
    # Check if API key is available
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("GEMINI_API_KEY"):
        print("Skipping convenience function test (no API key)")
        return
    
    provider = "openai" if os.getenv("OPENAI_API_KEY") else "gemini"
    
    try:
        print(f"Using convenience function with {provider}...")
        print()
        
        texts = ["Hello world", "Goodbye world"]
        embeddings = generate_embeddings(texts, provider=provider)
        
        print(f"✓ Generated {len(embeddings)} embeddings")
        print(f"  Dimension: {len(embeddings[0])}")
        print()
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        print()


def main():
    """Run all tests."""
    print("\n")
    print("=" * 70)
    print("EMBEDDINGS GENERATOR TEST SUITE")
    print("=" * 70)
    print("\n")
    
    try:
        # Test 1: Provider info
        generator = test_provider_info()
        
        if generator:
            # Test 2: Single embedding
            test_single_embedding(generator)
            
            # Test 3: Batch embeddings
            embeddings = test_batch_embeddings(generator)
            
            # Test 4: Similarity demo
            test_similarity_demo(embeddings)
        
        # Test 5: Convenience function
        test_convenience_function()
        
        print("=" * 70)
        print("✓ TEST SUITE COMPLETED")
        print("=" * 70)
        
    except Exception as e:
        print("=" * 70)
        print(f"✗ TEST SUITE FAILED: {e}")
        print("=" * 70)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
