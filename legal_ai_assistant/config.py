# config.py
"""
Configuration centralisée pour le chatbot FAQ
"""

# Configuration du modèle LLM - Optimisée pour CPU uniquement
LLM_CONFIG = {
    "model_name": "llama3.2:3b",
    "size": "~2.0GB (Ollama optimized)",
    "ram_required": "3-5GB",  # RAM nécessaire pour Llama 3.2 3B
    "description": "Llama 3.2 3B - Fast CPU-optimized inference with tool calling support",
    "max_new_tokens": 200,  # Plus de tokens pour Llama 3.2 3B
    "temperature": 0.1,
    "context_window": 4096,  # Contexte plus large pour Llama 3.2 3B
    "num_threads": 6,        # Plus de threads CPU
    "batch_size": 256,       # Batch plus petit pour CPU
    "quantization": {
        "enabled": False,  # Ollama handles optimization
        "load_in_4bit": False,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": False,
    },
    "cpu_optimization": {
        "enabled": False,  # Ollama handles optimization
        "use_cache": False,
        "low_cpu_mem_usage": False,
    }
}

# Configuration de l'embedding
EMBEDDING_CONFIG = {
    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "batch_size": 32,
    "normalize_embeddings": True,
    "show_progress": True,
}

# Configuration du vectorstore - Optimisée pour Llama 3.2 3B
VECTORSTORE_CONFIG = {
    "collection_name": "faq_docs",
    "persist_directory": "./chroma_db",
    "default_top_k": 3,  # Trois documents pour Llama 3.2 3B
    "max_context_length": 1200,  # Contexte plus large pour Llama 3.2 3B
}

# Configuration de l'application
APP_CONFIG = {
    "title": "French AI Law Guru",
    "page_layout": "wide",
    "initial_sidebar_state": "expanded",
    "max_top_k": 5,
    "default_top_k": 3,
}
