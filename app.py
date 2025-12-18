"""
AI Tutor Application Entry Point

Launches the AI Tutor web interface using Streamlit.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Run the AI Tutor application."""
    print("=" * 60)
    print("🎓 AI Tutor System - Starting...")
    print("=" * 60)
    
    # Get the path to the UI script
    ui_path = Path(__file__).parent / "ui" / "web_app.py"
    
    if not ui_path.exists():
        print(f"❌ Error: UI script not found at {ui_path}")
        return
    
    # Check for Gemini API Key
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("\n⚠️  WARNING: GEMINI_API_KEY not found in environment.")
        print("   Offline mode will work, but Embeddings generation requires an API key.")
        print("   Set it in your .env file or environment variables.\n")
    
    print("\n🚀 Launching Streamlit UI...")
    print("   Press Ctrl+C to stop the application.\n")
    
    # Run Streamlit
    try:
        # Use sys.executable to run streamlit as a module
        # This avoids issues where the 'streamlit' command is not in the PATH
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(ui_path)], check=True)
    except KeyboardInterrupt:
        print("\n👋 AI Tutor stopped.")
    except Exception as e:
        print(f"\n❌ Failed to launch UI: {e}")

if __name__ == "__main__":
    main()
