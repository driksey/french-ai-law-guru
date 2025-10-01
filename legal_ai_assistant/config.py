# config.py
"""
Centralized configuration for the French AI Law Assistant.

This module contains all configuration settings for:
- LLM models (Gemma 2 2B and Gemma 3 270M)
- Embedding models (multilingual support)
- Vectorstore settings
- Application settings
- Performance optimization parameters
"""

# =============================================================================
# LLM MODEL CONFIGURATIONS
# =============================================================================

# Primary model (Gemma 2 2B) - Optimized for CPU inference
LLM_CONFIG = {
    "model_name": "gemma2:2b",
    "size": "~1.6GB (Ollama optimized)",
    "ram_required": "≈3GB",
    "description": "Gemma 2 2B - Optimized for CPU inference",
    # Text generation parameters
    "max_new_tokens": 600,  # Increased for comprehensive legal analysis
    "temperature": 0.1,  # Low temperature for deterministic output
    "top_p": 0.8,  # Balanced for quality and speed
    "repeat_penalty": 1.05,  # Slightly increased for better quality
    # Context configuration
    "context_window": 3072,  # Increased for comprehensive legal analysis
    "use_cache": True,  # Enables caching of intermediate calculations
    # CPU optimizations
    "num_threads": 6,  # Optimized for multi-core CPU
    "batch_size": 128,  # Increased batch size for better throughput
    # Ollama-specific parameters
    "num_ctx": 3072,  # Increased context size for comprehensive analysis
    "num_gpu": 0,  # Number of GPUs used (0 = CPU only)
    "top_k": 10,  # Balanced for quality and speed
    "num_batch": 64,  # Increased batch size for better performance
    "use_mmap": True,  # Uses memory mapping to load the model
    "use_mlock": True,  # Locks the model in memory for better performance
    "low_vram": False,  # Disabled for systems with sufficient RAM
    "num_keep": 512,  # Increased tokens to keep for better context
    "tfs_z": 0.8,  # Balanced Tail Free Sampling
    "typical_p": 0.7,  # Balanced Typical Sampling
    # Quantization configuration
    "quantization": {
        "enabled": True,
        "method": "q4_K_M",  # Good balance of quality and speed
    },
    # CPU optimizations
    "cpu_optimization": {
        "enabled": True,
        "use_cache": True,
        "low_cpu_mem_usage": False,  # Disabled for systems with sufficient RAM
        "cpu_threads": 6,  # Optimized for multi-core CPU
        "memory_pool_size": 8192,  # Memory pool for better performance
    },
}

# Tool model (Gemma 3 270M) - Optimized for fast tool calls and analysis
QUESTION_ANALYSIS_CONFIG = {
    "model_name": "gemma3:270m",
    "size": "~270MB (Ollama optimized)",
    "ram_required": "≈1GB",
    "description": "Gemma 3 270M - Optimized for fast tool calls and analysis",
    # Text generation parameters
    "max_new_tokens": 150,  # Increased for better tool call generation
    "temperature": 0.1,  # Low temperature for deterministic output
    "top_p": 0.8,  # Balanced for quality and speed
    "repeat_penalty": 1.05,  # Slightly increased for better quality
    # Context configuration
    "context_window": 1024,  # Increased for better tool call context
    "use_cache": True,  # Enables caching
    # CPU optimizations
    "num_threads": 4,  # Optimized for lightweight model
    "batch_size": 64,  # Increased batch size for better throughput
    # Ollama-specific parameters
    "num_ctx": 1024,  # Increased context size for better tool calls
    "num_gpu": 0,  # Number of GPUs used (0 = CPU only)
    "top_k": 10,  # Balanced for quality and speed
    "num_batch": 32,  # Increased batch size for better performance
    "use_mmap": True,  # Uses memory mapping to load the model
    "use_mlock": True,  # Locks the model in memory for better performance
    "low_vram": False,  # Disabled for systems with sufficient RAM
    "num_keep": 128,  # Increased tokens to keep for better context
    "tfs_z": 0.8,  # Balanced Tail Free Sampling
    "typical_p": 0.7,  # Balanced Typical Sampling
}

# =============================================================================
# EMBEDDING AND VECTORSTORE CONFIGURATIONS
# =============================================================================

# Multilingual embedding model configuration
EMBEDDING_CONFIG = {
    "model_name": "sentence-transformers/distiluse-base-multilingual-cased",
    "batch_size": 64,  # Increased batch size for better throughput
    "normalize_embeddings": True,  # Normalizes embedding vectors
    "show_progress": True,  # Shows progress bar during loading
    "description": "Multilingual model optimized for CPU inference",
    "supported_languages": [
        "fr",
        "en",
        "de",
        "es",
        "it",
        "pt",
        "nl",
        "pl",
        "ru",
        "tr",
        "ar",
        "zh",
        "ja",
        "ko",
        "hi",
    ],
    "device": "cpu",  # Explicitly use CPU
    "max_seq_length": 512,  # Optimized sequence length
}

# Vector database configuration
VECTORSTORE_CONFIG = {
    "collection_name": "legal_docs",  # Collection name in ChromaDB
    "persist_directory": "./chroma_db",  # Persistent storage directory
    "max_context_length": 1200,  # Increased for comprehensive legal analysis
    "distance_metric": "cosine",  # Cosine similarity for better legal document matching
    "embedding_function": "sentence-transformers",  # Explicit embedding function
}

# =============================================================================
# APPLICATION CONFIGURATION
# =============================================================================

# User interface configuration
APP_CONFIG = {
    "title": "French AI Law Assistant",  # Application title
    "page_layout": "wide",  # Streamlit page layout
    "initial_sidebar_state": "expanded",  # Initial sidebar state
    "max_top_k": 6,  # Maximum documents for comprehensive analysis
    "default_top_k": 4,  # Default documents for better legal analysis
}
