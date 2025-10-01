# local_models.py
from langchain_ollama import ChatOllama

from .config import LLM_CONFIG


def create_local_llm(config=None):
    """
    Create the configured LLM model using Ollama.
    Inspired by: https://github.com/kaymen99/local-rag-researcher-deepseek/tree/main

    Args:
        config: Optional configuration dict. If None, uses LLM_CONFIG.

    Returns:
        ChatOllama client
    """
    if config is None:
        config = LLM_CONFIG

    model_name = config["model_name"]
    try:
        # Create ChatOllama instance with all parameters from config
        chat_model = ChatOllama(
            model=model_name,
            temperature=config["temperature"],
            num_predict=config["max_new_tokens"],
            verbose=True,
            # All parameters from config
            num_ctx=config["num_ctx"],
            num_thread=config["num_threads"],
            num_gpu=config["num_gpu"],
            repeat_penalty=config["repeat_penalty"],
            top_k=config["top_k"],
            top_p=config["top_p"],
            num_batch=config["num_batch"],
            use_mmap=config["use_mmap"],
            use_mlock=config["use_mlock"],
            low_vram=config["low_vram"],
            num_keep=config["num_keep"],
            tfs_z=config["tfs_z"],
            typical_p=config["typical_p"],
        )

        print("OK Ollama ChatOllama initialized successfully")
        print(f"   Model: {model_name}")
        print(f"   Temperature: {config['temperature']}")
        print(f"   Max tokens: {config['max_new_tokens']}")
        print(f"   Description: {config['description']}")

        return chat_model

    except Exception as e:
        print(f"ERROR Error creating Ollama client: {e}")
        print(f"Error type: {type(e).__name__}")
        print(f"Make sure Ollama is running and the model '{model_name}' is installed.")
        print(f"To install the model, run: ollama pull {model_name}")
        import traceback
        traceback.print_exc()
        return None


def get_model_info():
    """Get information about the configured model."""
    return {
        "size": LLM_CONFIG["size"],
        "ram_required": LLM_CONFIG["ram_required"],
        "description": LLM_CONFIG["description"]
    }


def get_embedding_info():
    """Get information about the configured embedding model."""
    from .config import EMBEDDING_CONFIG
    return {
        "size": "~135MB",
        "languages_count": len(EMBEDDING_CONFIG["supported_languages"]),
        "supported_languages": EMBEDDING_CONFIG["supported_languages"],
        "description": EMBEDDING_CONFIG["description"]
    }
