"""
Test Offline Teaching Mode & Caching

This script verifies:
1. Deterministic offline lecture generation (Intro/Body/Summary structure)
2. Lecture caching (file creation and reuse)
3. Fallback logic (forcing LLM failure -> Offline mode)

Usage:
    python test_offline_mode.py
"""

import sys
import shutil
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Load .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

from vector_store.database import VectorDatabase
from vector_store.embeddings import EmbeddingGenerator
from teaching.rag_engine import RAGTeachingEngine

def test_offline_mode():
    print("\n" + "="*80)
    print("TEST: OFFLINE TEACHING & CACHING")
    print("="*80)
    
    # 1. Setup
    cache_dir = Path("./test_lectures_cache")
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        
    db = VectorDatabase(collection_name="biology_textbook", persist_directory="./data/chroma_db")
    if db.get_document_count() == 0:
        print("❌ Database empty! Run 'test_rag_teaching.py' first to populate data.")
        return False
        
    embedder = EmbeddingGenerator(provider="gemini")
    
    # 2. Initialize in OFFLINE mode
    engine = RAGTeachingEngine(
        vector_database=db,
        embedding_generator=embedder,
        teaching_mode="OFFLINE",
        cache_dir=str(cache_dir)
    )
    
    topic = "photosynthesis"
    print(f"\n[1] Teaching '{topic}' in OFFLINE mode...")
    
    # 3. Generate Lecture (First time)
    lecture1 = engine.teach(topic)
    
    print("\nLecture Output Preview:")
    print("-" * 40)
    print(lecture1[:500] + "...")
    print("-" * 40)
    
    if "**Introduction**" not in lecture1 or "*Offline Mode:" not in lecture1:
        print("❌ Failed: Output format incorrect")
        return False
        
    # Verify Cache
    cache_file = cache_dir / "photosynthesis.txt"
    if not cache_file.exists():
        print(f"❌ Failed: Cache file not created at {cache_file}")
        return False
        
    print(f"✓ Lecture generated and cached at {cache_file}")
    
    # 4. Verify Caching (Second time)
    print(f"\n[2] Teaching '{topic}' again (should hit cache)...")
    lecture2 = engine.teach(topic)
    
    if lecture1 == lecture2:
        print("✓ Cache hit successful (Outputs match)")
    else:
        print("❌ Failed: Cache mismatch")
        return False
        
    # 5. Test Fallback (LLM Mode with invalid setup)
    print(f"\n[3] Testing LLM Fallback (simulated failure)...")
    
    # Force failure by passing invalid model/key implicitly or just checking manual logic
    # We'll use a new engine instance with LLM mode but simulate an error if possible, 
    # or just rely on the fact that if we mess up the key it falls back.
    # Actually, RAGTeachingEngine falls back during init or generation.
    
    # Let's try to init with LLM mode. If key exists, it will work. 
    # To test fallback, we can temporarily unset the key, but that affects embeddings too.
    # Instead, we will Mock the LLM client to raise an exception during generation.
    
    engine_llm = RAGTeachingEngine(
        vector_database=db,
        embedding_generator=embedder,
        teaching_mode="LLM",
        cache_dir=str(cache_dir)
    )
    
    # Sabotage the LLM client
    class BrokenLLM:
        def generate_content(self, prompt):
            raise Exception("Simulated API outage")
        class chat:
            class completions:
                def create(*args, **kwargs):
                    raise Exception("Simulated API outage")
                    
    engine_llm.llm_client = BrokenLLM()
    if engine_llm.llm_provider == "openai":
         engine_llm.llm_client = BrokenLLM() # Monkey patch for openai too structure wise
    
    topic_fallback = "cellular respiration"
    print(f"Teaching '{topic_fallback}' with broken LLM...")
    
    lecture_fb = engine_llm.teach(topic_fallback)
    
    if "*Offline Mode:" in lecture_fb:
        print("✓ Fallback successful! (Returned offline lecture)")
    else:
        print("❌ Failed: Did not fall back to offline mode")
        print("Output was:", lecture_fb[:100])
        return False

    print("\n" + "="*80)
    print("✓ ALL TESTS PASSED")
    print("="*80)
    
    # Cleanup
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        
    return True

if __name__ == "__main__":
    test_offline_mode()
