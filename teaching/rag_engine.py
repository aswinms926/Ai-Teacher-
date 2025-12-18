"""
RAG Teaching Engine

Implements Retrieval-Augmented Generation for teaching topics from textbook content.

WHAT IS RAG (Retrieval-Augmented Generation)?
- RAG combines retrieval (finding relevant information) with generation (creating responses)
- Instead of relying on the LLM's training data, RAG retrieves specific content first
- The LLM then uses ONLY this retrieved content to generate responses
- This grounds the AI in factual, source-based information

HOW RAG PREVENTS HALLUCINATIONS:
- Traditional LLMs can "hallucinate" - generate plausible but incorrect information
- RAG constrains the LLM to use ONLY the retrieved textbook content
- If the content doesn't exist in the textbook, the system admits it
- This ensures accuracy and trustworthiness in educational contexts

HOW THIS DIFFERS FROM A CHATBOT:
- Chatbot: Conversational, asks questions, maintains dialogue state
- Lecture Engine: One-way teaching, structured explanations, no back-and-forth
- Chatbot: May use general knowledge
- Lecture Engine: ONLY uses provided textbook content
- Chatbot: Flexible format
- Lecture Engine: Consistent lecture structure (intro, explanation, key points, summary)

TEACHING FLOW:
1. Student requests a topic (e.g., "photosynthesis")
2. Convert topic to embedding
3. Retrieve relevant chunks from vector database
4. Combine chunks into context
5. Generate structured lecture using Gemini
6. Return lecture to student
"""

import logging
import os
from typing import Optional, List, Tuple, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RAGTeachingEngine:
    """
    AI Teaching Engine using Retrieval-Augmented Generation.
    
    This engine retrieves relevant content from a textbook (stored in a vector database)
    and uses an LLM (Gemini) to generate structured, lecture-style explanations.
    
    The teaching is grounded in the textbook content, preventing hallucinations
    and ensuring accuracy.
    """
    
    def __init__(
        self,
        vector_database,
        embedding_generator,
        llm_provider: str = "gemini",
        llm_model: Optional[str] = None,
        top_k: int = 6
    ):
        """
        Initialize the RAG Teaching Engine.
        
        Args:
            vector_database: VectorDatabase instance with stored textbook chunks
            embedding_generator: EmbeddingGenerator instance for query embeddings
            llm_provider: LLM provider to use ("gemini" or "openai")
            llm_model: Specific model name (uses defaults if None)
            top_k: Number of relevant chunks to retrieve (default: 6)
        
        Raises:
            ValueError: If LLM provider is not supported
            ImportError: If required LLM library is not installed
        """
        self.vector_db = vector_database
        self.embedding_gen = embedding_generator
        self.llm_provider = llm_provider.lower()
        self.top_k = top_k
        
        # Set default models
        if llm_model is None:
            if self.llm_provider == "gemini":
                llm_model = "models/gemini-pro-latest"  # Exact model ID from list_models()
            elif self.llm_provider == "openai":
                llm_model = "gpt-3.5-turbo"
            else:
                raise ValueError(
                    f"Unsupported LLM provider: {llm_provider}. "
                    f"Supported: gemini, openai"
                )
        
        self.llm_model = llm_model
        
        # Initialize LLM client
        self.llm_client = None
        self._initialize_llm()
        
        logger.info(
            f"RAGTeachingEngine initialized: llm={self.llm_provider}/{self.llm_model}, "
            f"top_k={self.top_k}"
        )
    
    def _initialize_llm(self):
        """
        Initialize the LLM client based on the provider.
        
        This method lazy-loads the required library to avoid importing
        unnecessary dependencies.
        """
        try:
            if self.llm_provider == "gemini":
                # Import Google Generative AI library
                try:
                    import google.generativeai as genai
                except ImportError:
                    raise ImportError(
                        "Google Generative AI library not installed. "
                        "Install with: pip install google-generativeai"
                    )
                
                # Get API key
                api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
                if not api_key:
                    raise ValueError(
                        "Gemini API key not found. Set GEMINI_API_KEY or GOOGLE_API_KEY "
                        "environment variable."
                    )
                
                # Configure Gemini
                genai.configure(api_key=api_key)
                self.llm_client = genai.GenerativeModel(self.llm_model)
                logger.info("Gemini LLM client initialized successfully")
                
            elif self.llm_provider == "openai":
                # Import OpenAI library
                try:
                    from openai import OpenAI
                except ImportError:
                    raise ImportError(
                        "OpenAI library not installed. Install with: pip install openai"
                    )
                
                # Get API key
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError(
                        "OpenAI API key not found. Set OPENAI_API_KEY environment variable."
                    )
                
                self.llm_client = OpenAI(api_key=api_key)
                logger.info("OpenAI LLM client initialized successfully")
                
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {e}")
            raise
    
    def _retrieve_context(self, topic: str) -> Tuple[str, List[Tuple[str, float, Dict]]]:
        """
        Retrieve relevant content from the vector database.
        
        This is the "Retrieval" part of RAG. We convert the topic to an embedding
        and search for the most similar chunks in the textbook.
        
        Args:
            topic: The topic to teach
        
        Returns:
            Tuple of (combined_context, search_results)
        
        Raises:
            Exception: If retrieval fails
        """
        logger.info(f"Retrieving context for topic: '{topic}'")
        
        try:
            # Step 1: Convert topic to embedding
            query_embedding = self.embedding_gen.generate_embedding(topic)
            
            if not query_embedding:
                raise ValueError("Failed to generate embedding for topic")
            
            # Step 2: Search vector database for similar chunks
            results = self.vector_db.similarity_search(
                query_embedding,
                top_k=self.top_k
            )
            
            if not results:
                logger.warning(f"No relevant content found for topic: '{topic}'")
                return "", []
            
            # Step 3: Combine retrieved chunks into context
            # Sort by similarity score (highest first) - already sorted by ChromaDB
            context_chunks = []
            for text, score, metadata in results:
                context_chunks.append(text)
                logger.debug(f"Retrieved chunk (score={score:.4f}): {text[:100]}...")
            
            # Combine all chunks with separators
            combined_context = "\n\n---\n\n".join(context_chunks)
            
            logger.info(
                f"Retrieved {len(results)} chunks, "
                f"total context length: {len(combined_context)} chars"
            )
            
            return combined_context, results
            
        except Exception as e:
            logger.error(f"Context retrieval failed: {e}")
            raise
    
    def _generate_lecture(self, topic: str, context: str) -> str:
        """
        Generate a structured lecture using the LLM.
        
        This is the "Generation" part of RAG. We provide the retrieved context
        to the LLM and ask it to create a lecture-style explanation.
        
        The prompt is carefully designed to:
        - Use ONLY the provided context
        - Follow a structured format
        - Use simple, clear language
        - Avoid hallucinations
        
        Args:
            topic: The topic to teach
            context: Retrieved textbook content
        
        Returns:
            Generated lecture text
        
        Raises:
            Exception: If generation fails
        """
        logger.info(f"Generating lecture for topic: '{topic}'")
        
        # Construct the teaching prompt
        # This prompt is crucial for RAG - it constrains the LLM to use only the context
        prompt = f"""You are a school teacher explaining a topic from a textbook.

TOPIC: {topic}

TEXTBOOK CONTENT:
{context}

INSTRUCTIONS:
1. Use ONLY the information provided in the textbook content above
2. If the textbook content doesn't contain enough information about the topic, say so
3. Do NOT add information from your general knowledge
4. Do NOT make up facts or examples not in the textbook
5. Explain in simple, clear language suitable for students

LECTURE STRUCTURE:
Follow this exact structure:

**Introduction**
[Brief introduction to the topic - 1-2 sentences]

**Explanation**
[Main explanation of the topic using the textbook content - 3-5 paragraphs]

**Key Points**
• [Key point 1]
• [Key point 2]
• [Key point 3]
[Add more key points as needed from the textbook]

**Summary**
[Brief summary - 2-3 sentences]

Now, generate the lecture:"""

        try:
            if self.llm_provider == "gemini":
                # Generate with Gemini
                response = self.llm_client.generate_content(prompt)
                lecture = response.text
                
            elif self.llm_provider == "openai":
                # Generate with OpenAI
                response = self.llm_client.chat.completions.create(
                    model=self.llm_model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful school teacher who explains topics clearly and accurately using only the provided textbook content."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.3,  # Lower temperature for more factual responses
                    max_tokens=1000
                )
                lecture = response.choices[0].message.content
            
            logger.info(f"Generated lecture ({len(lecture)} chars)")
            logger.debug(f"Lecture preview: {lecture[:200]}...")
            
            return lecture
            
        except Exception as e:
            logger.error(f"Lecture generation failed: {e}")
            raise
    
    def teach(self, topic: str) -> str:
        """
        Teach a topic using RAG.
        
        This is the main method that orchestrates the entire RAG pipeline:
        1. Retrieve relevant content from the textbook
        2. Generate a structured lecture using the LLM
        3. Return the lecture to the student
        
        Args:
            topic: The topic to teach (e.g., "photosynthesis", "cell division")
        
        Returns:
            Structured lecture text
        
        Example:
            >>> engine = RAGTeachingEngine(vector_db, embedding_gen)
            >>> lecture = engine.teach("photosynthesis")
            >>> print(lecture)
            **Introduction**
            Photosynthesis is the process by which plants...
        """
        if not topic or not topic.strip():
            logger.warning("Empty topic provided")
            return "Please provide a topic to teach."
        
        logger.info(f"Teaching request: '{topic}'")
        
        try:
            # Step 1: Retrieve relevant context (RAG - Retrieval)
            context, results = self._retrieve_context(topic)
            
            # Step 2: Check if we have sufficient context
            if not context or len(context) < 100:
                logger.warning(f"Insufficient content for topic: '{topic}'")
                return (
                    "The provided material does not contain enough information "
                    "to teach this topic."
                )
            
            # Step 3: Generate lecture (RAG - Generation)
            lecture = self._generate_lecture(topic, context)
            
            # Step 4: Add metadata footer (optional)
            # This helps students know the lecture is based on the textbook
            footer = f"\n\n---\n*This lecture is based on {len(results)} relevant sections from your textbook.*"
            
            return lecture + footer
            
        except Exception as e:
            logger.error(f"Teaching failed for topic '{topic}': {e}")
            return (
                f"I encountered an error while preparing the lecture on '{topic}'. "
                f"Please try again or choose a different topic."
            )
    
    def get_engine_info(self) -> Dict[str, Any]:
        """
        Get information about the teaching engine configuration.
        
        Returns:
            Dictionary with engine details
        """
        return {
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "top_k": self.top_k,
            "vector_db_documents": self.vector_db.get_document_count(),
            "embedding_provider": self.embedding_gen.provider,
            "embedding_model": self.embedding_gen.model_name
        }


# Convenience function for simple usage
def create_teaching_engine(
    vector_database,
    embedding_generator,
    llm_provider: str = "gemini",
    top_k: int = 6
):
    """
    Convenience function to create a RAG teaching engine.
    
    Args:
        vector_database: VectorDatabase instance
        embedding_generator: EmbeddingGenerator instance
        llm_provider: LLM provider ("gemini" or "openai")
        top_k: Number of chunks to retrieve
    
    Returns:
        RAGTeachingEngine instance
    
    Example:
        >>> from teaching.rag_engine import create_teaching_engine
        >>> engine = create_teaching_engine(db, embedder)
        >>> lecture = engine.teach("evolution")
    """
    return RAGTeachingEngine(
        vector_database=vector_database,
        embedding_generator=embedding_generator,
        llm_provider=llm_provider,
        top_k=top_k
    )
