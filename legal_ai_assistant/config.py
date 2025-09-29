# config.py
"""
Configuration centralisée pour le chatbot FAQ
"""

# Configuration du modèle LLM
LLM_CONFIG = {
    "model_name": "llama3.1:8b",
    "size": "~4.9GB (Ollama optimized)",
    "ram_required": "6-8GB",
    "description": "Llama 3.1 8B - Excellent model with full tool calling support for RAG applications",
    "max_new_tokens": 200,
    "temperature": 0.1,
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

# Configuration du vectorstore
VECTORSTORE_CONFIG = {
    "collection_name": "faq_docs",
    "persist_directory": "./chroma_db",
    "default_top_k": 3,
}

# Configuration de l'application
APP_CONFIG = {
    "title": "French AI Law Guru",
    "page_layout": "wide",
    "initial_sidebar_state": "expanded",
    "max_top_k": 5,
    "default_top_k": 3,
}
