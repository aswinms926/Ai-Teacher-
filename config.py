"""
Configuration File

Central configuration for the AI Tutor system.

Contains:
- API keys and credentials
- Model settings
- Database configuration
- UI preferences
"""


class Config:
    """
    Configuration settings for the AI Tutor system.
    
    TODO:
    - Add environment variable loading
    - Add validation for required settings
    - Support multiple environments (dev, prod)
    """
    
    # Vector Store Settings
    VECTOR_DB_TYPE = "chromadb"  # Options: chromadb, pinecone, faiss
    VECTOR_DB_PATH = "./data/chroma_db"
    
    # Embedding Settings
    EMBEDDING_PROVIDER = "gemini"  # Options: openai, gemini
    EMBEDDING_MODEL = "text-embedding-004"  # Gemini embedding model
    EMBEDDING_DIMENSION = 768  # Gemini: 768, OpenAI: 1536
    
    # LLM Settings
    LLM_PROVIDER = "openai"  # Options: openai, anthropic, gemini
    LLM_MODEL = "gpt-3.5-turbo"
    LLM_TEMPERATURE = 0.7
    LLM_MAX_TOKENS = 500
    
    # API Keys (Load from environment variables)
    import os
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", "")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    
    # Text Processing Settings
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    
    # Retrieval Settings
    TOP_K_DOCUMENTS = 3
    SIMILARITY_THRESHOLD = 0.7
    
    # UI Settings
    UI_TYPE = "cli"  # Options: cli, streamlit, gradio
    WEB_PORT = 8501
    
    # Data Paths
    DATA_DIR = "./data"
    DOCUMENTS_DIR = "./data/documents"
    
    @classmethod
    def load_from_env(cls):
        """
        Load configuration from environment variables.
        
        TODO: Implement environment variable loading
        """
        pass
    
    @classmethod
    def validate(cls):
        """
        Validate configuration settings.
        
        TODO: Implement validation logic
        """
        pass
