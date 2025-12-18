"""
List available Gemini models to find the correct one.

Usage:
    python list_gemini_models.py
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


def list_models():
    """List all available Gemini models."""
    
    print("=" * 70)
    print("AVAILABLE GEMINI MODELS")
    print("=" * 70)
    print()
    
    # Check API key
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("✗ Gemini API key not found")
        print("Set: $env:GEMINI_API_KEY=\"your-key\"")
        return False
    
    print(f"✓ API key found")
    print()
    
    try:
        import google.generativeai as genai
        
        # Configure
        genai.configure(api_key=api_key)
        
        print("Fetching available models...")
        print()
        
        # List models
        models = genai.list_models()
        
        print("Available models for generateContent:")
        print("-" * 70)
        
        for model in models:
            if 'generateContent' in model.supported_generation_methods:
                print(f"  • {model.name}")
                print(f"    Display name: {model.display_name}")
                print(f"    Description: {model.description[:100]}...")
                print()
        
        print("-" * 70)
        print()
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to list models: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    list_models()
