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
            # CPU-specific optimizations for Llama 3.2 3B
            num_ctx=4096,  # Larger context for Llama 3.2 3B
            num_thread=6,  # Use more CPU threads for better performance
            num_gpu=0,     # Disable GPU completely
            repeat_penalty=1.1,
            top_k=40,
            top_p=0.9,
            # CPU memory optimizations for Llama 3.2 3B
            num_batch=512,  # Larger batch size for Llama 3.2 3B
            use_mmap=True,  # Memory mapping for faster loading
            use_mlock=False, # Disable mlock on CPU to avoid memory issues
            low_vram=True,  # Optimize for lower memory usage
            # Additional CPU optimizations
            num_keep=5,     # Keep fewer tokens in context
            tfs_z=1.0,      # Tail free sampling for faster generation
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

