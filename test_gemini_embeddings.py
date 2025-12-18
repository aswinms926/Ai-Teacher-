"""
Gemini Embeddings Sanity Test

Quick test to verify that Gemini embeddings are working correctly.

Usage:
    # Option 1: Set environment variable
    $env:GEMINI_API_KEY="your-key-here"
    
    # Option 2: Create .env file with your key
    # (The script will automatically load it)
    
    # Then run:
    python test_gemini_embeddings.py
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Load .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()  # This loads the .env file
    print("✓ Loaded .env file")
except ImportError:
    print("ℹ python-dotenv not installed, using environment variables only")
except Exception:
    pass  # Silently continue if .env doesn't exist


def main():
    """Run a simple sanity test for Gemini embeddings."""
    
    print("=" * 70)
    print("GEMINI EMBEDDINGS SANITY TEST")
    print("=" * 70)
    print()
    
    # Step 1: Check if API key is set
    print("Step 1: Checking for Gemini API key...")
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        print("✗ FAILED: Gemini API key not found!")
        print()
        print("Please set your API key:")
        print("  PowerShell: $env:GEMINI_API_KEY=\"your-key-here\"")
        print()
        print("Get your key from: https://makersuite.google.com/app/apikey")
        print()
        sys.exit(1)
    
    print(f"✓ API key found: {api_key[:10]}...{api_key[-4:]}")
    print()
    
    # Step 2: Import and initialize
    print("Step 2: Initializing Gemini embedding generator...")
    try:
        from vector_store.embeddings import EmbeddingGenerator
        
        generator = EmbeddingGenerator(provider="gemini")
        print("✓ Generator initialized successfully")
        print()
        
    except ImportError as e:
        print(f"✗ FAILED: Missing dependency - {e}")
        print()
        print("Install required package:")
        print("  pip install google-generativeai")
        print()
        sys.exit(1)
        
    except ValueError as e:
        print(f"✗ FAILED: {e}")
        print()
        sys.exit(1)
        
    except Exception as e:
        print(f"✗ FAILED: Unexpected error - {e}")
        print()
        sys.exit(1)
    
    # Step 3: Generate embedding
    print("Step 3: Generating embedding for sample text...")
    sample_text = "Photosynthesis is the process by which plants make food."
    
    print(f"Sample text: '{sample_text}'")
    print()
    
    try:
        embedding = generator.generate_embedding(sample_text)
        
        if not embedding:
            print("✗ FAILED: Received empty embedding")
            print()
            sys.exit(1)
        
        print("✓ Embedding generated successfully!")
        print()
        
    except Exception as e:
        print(f"✗ FAILED: Error generating embedding - {e}")
        print()
        print("Common issues:")
        print("  - Invalid API key")
        print("  - Network connection problem")
        print("  - API quota exceeded")
        print()
        sys.exit(1)
    
    # Step 4: Display results
    print("Step 4: Verifying embedding...")
    print(f"  Embedding dimension: {len(embedding)}")
    print(f"  Expected dimension: 768 (Gemini text-embedding-004)")
    print(f"  First 5 values: {embedding[:5]}")
    print()
    
    if len(embedding) == 768:
        print("✓ Dimension matches expected value (768)")
    else:
        print(f"⚠ Warning: Unexpected dimension (got {len(embedding)}, expected 768)")
    
    print()
    
    # Final summary
    print("=" * 70)
    print("✓ SANITY TEST PASSED!")
    print("=" * 70)
    print()
    print("Summary:")
    print(f"  ✓ API key configured")
    print(f"  ✓ Gemini client initialized")
    print(f"  ✓ Embedding generated successfully")
    print(f"  ✓ Embedding dimension: {len(embedding)}")
    print()
    print("Your Gemini embeddings are working correctly! 🎉")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
