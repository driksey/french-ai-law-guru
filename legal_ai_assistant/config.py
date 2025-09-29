# config.py
"""
Configuration centralisée pour le chatbot FAQ
"""

# Configuration du modèle LLM - Optimisée pour CPU uniquement
LLM_CONFIG = {
    "model_name": "llama3.2:1b-instruct-q4_K_M",
    "size": "~1.1GB (quantized, Ollama optimized)",
    "ram_required": "≈2GB",  # Compatible CPU/RAM modeste
    "description": "Llama 3.2 1B Instruct - Quantized Q4_K_M for fast inference",

    # Génération
    "max_new_tokens": 300,   # Réponses concises, réduit coupures
    "temperature": 0.1,      # Réponses factuelles et stables
    "top_p": 0.9,            # Un peu de diversité mais contrôlée
    "repeat_penalty": 1.1,   # Évite les répétitions

    # Contexte
    "context_window": 1024,  # Plus rapide que 2048, suffisant pour RAG
    "use_cache": True,       # Active le cache pour accélérer la génération

    # CPU optimization
    "num_threads": 6,        # Ajuster selon tes cores CPU
    "batch_size": 128,       # Batch modeste → meilleur équilibre vitesse/mémoire

    # Paramètres Ollama spécifiques
    "num_ctx": 1024,         # Context window pour Ollama
    "num_gpu": 0,            # Désactiver GPU
    "top_k": 20,             # Réduit pour vitesse
    "num_batch": 128,        # Batch size aligné avec batch_size
    "use_mmap": True,        # Memory mapping pour chargement rapide
    "use_mlock": False,      # Désactiver mlock sur CPU
    "low_vram": True,        # Optimiser pour faible mémoire
    "num_keep": 1,           # Garder moins de tokens pour vitesse
    "tfs_z": 0.9,            # Tail free sampling réduit
    "typical_p": 0.8,        # Optimisation vitesse maximale

    # Quantization (handled by Ollama, mais documenté)
    "quantization": {
        "enabled": True,
        "method": "q4_K_M",  # Quantization spécifique déjà choisie
    },
    "cpu_optimization": {
        "enabled": True,     # CPU optimization activée
        "use_cache": True,   # Cache activé
        "low_cpu_mem_usage": True,
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
    "max_context_length": 400,  # Réduit pour vitesse maximale
}

# Configuration de l'application
APP_CONFIG = {
    "title": "French AI Law Assistant",
    "page_layout": "wide",
    "initial_sidebar_state": "expanded",
    "max_top_k": 3,
    "default_top_k": 2,
}
