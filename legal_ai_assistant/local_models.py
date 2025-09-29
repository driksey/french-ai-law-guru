# local_models.py
import os
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
        # Create ChatOllama instance
        chat_model = ChatOllama(
            model=model_name,
            temperature=LLM_CONFIG["temperature"],
            num_predict=LLM_CONFIG["max_new_tokens"],
            verbose=True
        )
        
        print(f"✅ Ollama ChatOllama initialized successfully")
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

