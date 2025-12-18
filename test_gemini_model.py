"""
Quick test to verify Gemini text generation model works.

Usage:
    python test_gemini_model.py
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Load .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass


def test_gemini_model():
    """Test that gemini-1.0-pro model works for text generation."""
    
    print("=" * 70)
    print("GEMINI TEXT GENERATION MODEL TEST")
    print("=" * 70)
    print()
    
    # Check API key
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("✗ Gemini API key not found")
        print("Set: $env:GEMINI_API_KEY=\"your-key\"")
        return False
    
    print(f"✓ API key found: {api_key[:10]}...{api_key[-4:]}")
    print()
    
    # Test Gemini model
    print("Testing models/gemini-pro-latest...")
    print()
    
    try:
        import google.generativeai as genai
        
        # Configure
        genai.configure(api_key=api_key)
        
        # Create model with exact ID
        model = genai.GenerativeModel("models/gemini-pro-latest")
        
        print("✓ Model initialized: models/gemini-pro-latest")
        print()
        
        # Test generation
        print("Generating test response...")
        response = model.generate_content("Explain photosynthesis in one sentence.")
        
        print("✓ Generation successful!")
        print()
        print("Response:")
        print("-" * 70)
        print(response.text)
        print("-" * 70)
        print()
        
        print("=" * 70)
        print("✓ TEST PASSED - models/gemini-pro-latest is working!")
        print("=" * 70)
        print()
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        print()
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_gemini_model()
    sys.exit(0 if success else 1)
