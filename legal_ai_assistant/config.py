# config.py
"""
Centralized configuration for the FAQ chatbot
"""

# LLM model configuration - Optimized for CPU only
LLM_CONFIG = {
    "model_name": "gemma2:2b",
    "size": "~1.6GB (Ollama optimized)",
    "ram_required": "≈3GB",
    "description": "Gemma 2 2B - Google's latest efficient model for fast inference",

    # Text generation parameters - Optimized for speed and token efficiency
    "max_new_tokens": 300,   # Further reduced for faster generation and less truncation risk
    "temperature": 0.1,      # Low temperature for deterministic output
    "top_p": 0.7,            # Further reduced for faster sampling
    "repeat_penalty": 1.02,  # Minimal penalty for speed

    # Context configuration - Optimized for token efficiency
    "context_window": 1536,  # Further reduced for faster processing and token savings
    "use_cache": True,       # Enables caching of intermediate calculations
    
    # CPU optimizations - Optimized for speed
    "num_threads": 8,        # Increased threads for better CPU utilization
    "batch_size": 64,        # Reduced batch size for faster processing
    
    # Ollama-specific parameters - Speed and token optimized
    "num_ctx": 1536,         # Further reduced context size for Ollama
    "num_gpu": 0,            # Number of GPUs used (0 = CPU only)
    "top_k": 8,              # Further reduced for faster sampling
    "num_batch": 32,         # Smaller batch size for faster processing
    "use_mmap": True,        # Uses memory mapping to load the model
    "use_mlock": False,      # Locks the model in memory (disabled for CPU)
    "low_vram": True,        # Memory saving mode
    "num_keep": 256,         # Minimal tokens to keep for speed
    "tfs_z": 0.7,            # Further reduced Tail Free Sampling for speed
    "typical_p": 0.6,        # Further reduced Typical Sampling for speed

    # Quantization configuration
    "quantization": {
        "enabled": True,
        "method": "q4_K_M",  # Quantization method used by Ollama
    },
    # CPU optimizations
    "cpu_optimization": {
        "enabled": True,
        "use_cache": True,
        "low_cpu_mem_usage": True,
    }
}

# Multilingual embedding model configuration
EMBEDDING_CONFIG = {
    "model_name": "sentence-transformers/distiluse-base-multilingual-cased",
    "batch_size": 32,        # Batch size for embedding processing
    "normalize_embeddings": True,  # Normalizes embedding vectors
    "show_progress": True,   # Shows progress bar during loading
    "description": "Multilingual model optimized for French-English",
    "supported_languages": ["fr", "en", "de", "es", "it", "pt", "nl", "pl", "ru", "tr", "ar", "zh", "ja", "ko", "hi"],
}

# Vector database configuration
VECTORSTORE_CONFIG = {
    "collection_name": "legal_docs",   # Collection name in ChromaDB
    "persist_directory": "./chroma_db", # Persistent storage directory
    "max_context_length": 400,         # Further reduced for token efficiency and speed
}

# User interface configuration
APP_CONFIG = {
    "title": "French AI Law Assistant",    # Application title
    "page_layout": "wide",                 # Streamlit page layout
    "initial_sidebar_state": "expanded",   # Initial sidebar state
    "max_top_k": 4,                        # Maximum number of documents to retrieve
    "default_top_k": 3,                    # Default number of documents to retrieve (optimized for quality)
}
