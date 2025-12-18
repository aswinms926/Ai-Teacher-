"""
Complete Ingestion Pipeline Test

This script demonstrates the complete ingestion pipeline:
1. Load PDF document
2. Clean and chunk text
3. Extract metadata
4. Display results

Usage:
    python test_complete_pipeline.py <path_to_pdf>
    
    Or use sample text if no PDF provided:
    python test_complete_pipeline.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from ingestion.document_loader import DocumentLoader
from ingestion.text_processor import TextProcessor


def test_with_pdf(pdf_path: str):
    """Test the complete pipeline with a real PDF."""
    print("=" * 80)
    print("COMPLETE INGESTION PIPELINE TEST - PDF MODE")
    print("=" * 80)
    print()
    
    # Step 1: Load PDF
    print("STEP 1: Loading PDF Document")
    print("-" * 80)
    
    loader = DocumentLoader()
    
    try:
        raw_text = loader.load_pdf(pdf_path)
        print(f"✓ PDF loaded successfully")
        print(f"  Raw text length: {len(raw_text)} characters")
        print(f"  Raw text words: {len(raw_text.split())} words")
        print()
    except Exception as e:
        print(f"✗ Failed to load PDF: {e}")
        return False
    
    # Step 2: Process and chunk text
    print("STEP 2: Processing and Chunking Text")
    print("-" * 80)
    
    processor = TextProcessor(chunk_size=600, chunk_overlap=60)
    
    # Clean text
    clean_text = processor.clean_text(raw_text)
    print(f"✓ Text cleaned")
    print(f"  Cleaned text length: {len(clean_text)} characters")
    print()
    
    # Chunk text
    chunks = processor.chunk_text(raw_text)
    print(f"✓ Text chunked")
    print(f"  Total chunks: {len(chunks)}")
    print(f"  Average chunk size: {sum(len(c) for c in chunks) // len(chunks) if chunks else 0} chars")
    print()
    
    # Step 3: Extract metadata
    print("STEP 3: Extracting Metadata")
    print("-" * 80)
    
    metadata = processor.extract_metadata(raw_text)
    print(f"✓ Metadata extracted")
    for key, value in metadata.items():
        if key == 'potential_headers':
            print(f"  {key}: {len(value)} headers found")
            for i, header in enumerate(value[:3], 1):
                print(f"    {i}. {header}")
        else:
            print(f"  {key}: {value}")
    print()
    
    # Step 4: Display sample chunks
    print("STEP 4: Sample Chunks Preview")
    print("-" * 80)
    
    num_samples = min(3, len(chunks))
    for i in range(num_samples):
        print(f"\nChunk {i + 1} of {len(chunks)} ({len(chunks[i])} chars):")
        print("─" * 80)
        preview = chunks[i][:300] + ("..." if len(chunks[i]) > 300 else "")
        print(preview)
    
    print()
    print("=" * 80)
    print("✓ PIPELINE TEST COMPLETED SUCCESSFULLY")
    print("=" * 80)
    
    return True


def test_with_sample_text():
    """Test the pipeline with sample educational text."""
    print("=" * 80)
    print("COMPLETE INGESTION PIPELINE TEST - SAMPLE TEXT MODE")
    print("=" * 80)
    print()
    
    # Sample educational text about AI
    sample_text = """
    Artificial Intelligence: A Comprehensive Introduction
    
    Artificial Intelligence (AI) has become one of the most transformative 
    technologies of the 21st century. From virtual assistants to autonomous 
    vehicles, AI systems are increasingly integrated into our daily lives.
    
    What is Artificial Intelligence?
    
    Artificial Intelligence refers to the simulation of human intelligence in 
    machines that are programmed to think and learn like humans. The term may 
    also be applied to any machine that exhibits traits associated with a human 
    mind, such as learning and problem-solving.
    
    History of AI
    
    The field of AI research was founded in 1956 at a conference at Dartmouth 
    College. Early AI researchers developed algorithms that could solve algebra 
    problems, prove geometric theorems, and learn to speak English. However, 
    progress was slower than expected, leading to several "AI winters" where 
    funding and interest declined.
    
    Machine Learning
    
    Machine learning is a subset of AI that focuses on building systems that 
    can learn from data. Instead of being explicitly programmed to perform a 
    task, these systems improve their performance over time through experience. 
    There are three main types of machine learning: supervised learning, 
    unsupervised learning, and reinforcement learning.
    
    Supervised Learning
    
    In supervised learning, the algorithm learns from labeled training data. 
    The system is provided with input-output pairs and learns to map inputs to 
    outputs. Common applications include image classification, spam detection, 
    and medical diagnosis.
    
    Deep Learning
    
    Deep learning is a specialized form of machine learning that uses neural 
    networks with multiple layers. These deep neural networks can automatically 
    learn hierarchical representations of data, making them particularly 
    effective for tasks like image recognition, natural language processing, 
    and speech recognition.
    
    Natural Language Processing
    
    Natural Language Processing (NLP) is a branch of AI that focuses on the 
    interaction between computers and human language. NLP enables machines to 
    read, understand, and derive meaning from human languages. Applications 
    include chatbots, translation services, and sentiment analysis.
    
    Computer Vision
    
    Computer vision is another important area of AI that enables machines to 
    interpret and understand visual information from the world. This technology 
    powers facial recognition systems, autonomous vehicles, medical image 
    analysis, and augmented reality applications.
    
    Ethics and Challenges
    
    As AI becomes more powerful and widespread, important ethical questions 
    arise. Issues include bias in AI systems, privacy concerns, job displacement, 
    and the potential for misuse. Researchers and policymakers are working to 
    develop frameworks for responsible AI development and deployment.
    
    The Future of AI
    
    The future of AI holds immense potential. Advances in quantum computing, 
    neuromorphic chips, and algorithmic improvements promise to make AI systems 
    even more capable. However, realizing this potential while addressing 
    ethical concerns will be crucial for ensuring that AI benefits all of humanity.
    
    Conclusion
    
    Artificial Intelligence is a rapidly evolving field with far-reaching 
    implications for society. Understanding its fundamentals, capabilities, 
    and limitations is essential for anyone looking to work with or be informed 
    about this transformative technology.
    """
    
    # Step 1: Simulate PDF loading
    print("STEP 1: Using Sample Text (simulating PDF load)")
    print("-" * 80)
    print(f"✓ Sample text loaded")
    print(f"  Text length: {len(sample_text)} characters")
    print(f"  Text words: {len(sample_text.split())} words")
    print()
    
    # Step 2: Process and chunk
    print("STEP 2: Processing and Chunking Text")
    print("-" * 80)
    
    processor = TextProcessor(chunk_size=600, chunk_overlap=60)
    
    chunks = processor.chunk_text(sample_text)
    print(f"✓ Text chunked")
    print(f"  Total chunks: {len(chunks)}")
    print(f"  Average chunk size: {sum(len(c) for c in chunks) // len(chunks)} chars")
    print()
    
    # Step 3: Extract metadata
    print("STEP 3: Extracting Metadata")
    print("-" * 80)
    
    metadata = processor.extract_metadata(sample_text)
    print(f"✓ Metadata extracted")
    for key, value in metadata.items():
        if key == 'potential_headers':
            print(f"  {key}: {len(value)} headers found")
            for i, header in enumerate(value, 1):
                print(f"    {i}. {header}")
        else:
            print(f"  {key}: {value}")
    print()
    
    # Step 4: Display all chunks
    print("STEP 4: All Chunks Preview")
    print("-" * 80)
    
    for i, chunk in enumerate(chunks, 1):
        print(f"\nChunk {i} of {len(chunks)} ({len(chunk)} chars):")
        print("─" * 80)
        # Show first 250 chars of each chunk
        preview = chunk[:250] + ("..." if len(chunk) > 250 else "")
        print(preview)
    
    print()
    
    # Step 5: Show chunk overlap
    print("\nSTEP 5: Demonstrating Chunk Overlap")
    print("-" * 80)
    
    if len(chunks) > 1:
        for i in range(min(2, len(chunks) - 1)):
            print(f"\nOverlap between Chunk {i + 1} and Chunk {i + 2}:")
            chunk1_end = chunks[i][-80:]
            chunk2_start = chunks[i + 1][:80]
            print(f"  Chunk {i + 1} ends: ...{chunk1_end}")
            print(f"  Chunk {i + 2} starts: {chunk2_start}...")
            print()
    
    print("=" * 80)
    print("✓ PIPELINE TEST COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print()
    
    # Summary
    print("SUMMARY")
    print("-" * 80)
    print(f"✓ Successfully processed {len(sample_text)} characters")
    print(f"✓ Created {len(chunks)} chunks ready for embedding")
    print(f"✓ Extracted metadata with {len(metadata['potential_headers'])} headers")
    print()
    print("NEXT STEPS:")
    print("  1. Generate embeddings for each chunk")
    print("  2. Store chunks and embeddings in vector database")
    print("  3. Implement retrieval for RAG queries")
    print("=" * 80)


def main():
    """Main function."""
    if len(sys.argv) > 1:
        # PDF mode
        pdf_path = sys.argv[1]
        success = test_with_pdf(pdf_path)
        sys.exit(0 if success else 1)
    else:
        # Sample text mode
        print("\nNo PDF provided. Using sample educational text.\n")
        print("To test with a PDF, run:")
        print("  python test_complete_pipeline.py path/to/your.pdf\n")
        test_with_sample_text()


if __name__ == "__main__":
    main()
