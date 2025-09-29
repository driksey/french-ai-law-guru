# config.py
"""
Configuration centralisée pour le chatbot FAQ
"""

# Configuration du modèle LLM - Optimisée pour CPU uniquement
LLM_CONFIG = {
    "model_name": "llama3.2:1b",
    "size": "~1.3GB (Ollama optimized)",
    "ram_required": "2-3GB",  # RAM nécessaire pour Llama 3.2 1B
    "description": "Llama 3.2 1B - Ultra-fast CPU-optimized inference with tool calling support",
    "max_new_tokens": 200,  # Plus de tokens pour Llama 3.2 1B
    "temperature": 0.1,
    "context_window": 2048,  # Réduit pour vitesse maximale
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

# Configuration du vectorstore - Optimisée pour Llama 3.2 1B
VECTORSTORE_CONFIG = {
    "collection_name": "faq_docs",
    "persist_directory": "./chroma_db",
    "max_context_length": 800,  # Réduit pour vitesse maximale
}

# Configuration de l'application
APP_CONFIG = {
    "title": "French AI Law Guru",
    "page_layout": "wide",
    "initial_sidebar_state": "expanded",
    "max_top_k": 5,
    "default_top_k": 3,
}
