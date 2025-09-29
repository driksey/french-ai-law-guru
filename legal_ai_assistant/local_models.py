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
        # Create ChatOllama instance optimized for CPU-only systems
        chat_model = ChatOllama(
            model=model_name,
            temperature=LLM_CONFIG["temperature"],
            num_predict=LLM_CONFIG["max_new_tokens"],
            verbose=True,
            # CPU-specific optimizations for speed
            num_ctx=2048,  # Reduced context for speed
            num_thread=8,  # More threads for better performance
            num_gpu=0,     # Disable GPU completely
            repeat_penalty=1.05,  # Reduced for speed
            top_k=20,      # Reduced for speed
            top_p=0.8,     # Reduced for speed
            # CPU memory optimizations for speed
            num_batch=1024,  # Larger batch for speed
            use_mmap=True,   # Memory mapping for faster loading
            use_mlock=False, # Disable mlock on CPU to avoid memory issues
            low_vram=True,   # Optimize for lower memory usage
            # Additional CPU optimizations for speed
            num_keep=3,      # Keep fewer tokens for speed
            tfs_z=0.95,      # Tail free sampling for faster generation
            typical_p=0.9,   # Speed optimization
        )

        print("✅ Ollama ChatOllama initialized successfully")
        print(f"   Model: {model_name}")
        print(f"   Temperature: {LLM_CONFIG['temperature']}")
        print(f"   Max tokens: {LLM_CONFIG['max_new_tokens']}")
        print(f"   Description: {LLM_CONFIG['description']}")

        return chat_model

    except Exception as e:
        print(f"❌ Error creating Ollama client: {e}")
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

