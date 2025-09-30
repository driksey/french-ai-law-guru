# local_models.py
from langchain_ollama import ChatOllama

from .config import LLM_CONFIG


def create_local_llm():
    """
    Create the configured LLM model using Ollama.
    Inspired by: https://github.com/kaymen99/local-rag-researcher-deepseek/tree/main

    Returns:
        ChatOllama client
    """
    model_name = LLM_CONFIG["model_name"]
    try:
        # Create ChatOllama instance with all parameters from LLM_CONFIG
        chat_model = ChatOllama(
            model=model_name,
            temperature=LLM_CONFIG["temperature"],
            num_predict=LLM_CONFIG["max_new_tokens"],
            verbose=True,
            # All parameters from LLM_CONFIG
            num_ctx=LLM_CONFIG["num_ctx"],
            num_thread=LLM_CONFIG["num_threads"],
            num_gpu=LLM_CONFIG["num_gpu"],
            repeat_penalty=LLM_CONFIG["repeat_penalty"],
            top_k=LLM_CONFIG["top_k"],
            top_p=LLM_CONFIG["top_p"],
            num_batch=LLM_CONFIG["num_batch"],
            use_mmap=LLM_CONFIG["use_mmap"],
            use_mlock=LLM_CONFIG["use_mlock"],
            low_vram=LLM_CONFIG["low_vram"],
            num_keep=LLM_CONFIG["num_keep"],
            tfs_z=LLM_CONFIG["tfs_z"],
            typical_p=LLM_CONFIG["typical_p"],
        )

        print("OK Ollama ChatOllama initialized successfully")
        print(f"   Model: {model_name}")
        print(f"   Temperature: {LLM_CONFIG['temperature']}")
        print(f"   Max tokens: {LLM_CONFIG['max_new_tokens']}")
        print(f"   Description: {LLM_CONFIG['description']}")

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
