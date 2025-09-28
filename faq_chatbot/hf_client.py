# hf_client.py
import os
from dotenv import load_dotenv
from pathlib import Path
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# Charger le .env depuis la racine
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL = os.getenv("HF_MODEL", "HuggingFaceH4/zephyr-7b-beta")

if not HF_TOKEN:
    raise EnvironmentError("Set HF_TOKEN in environment (see .env.example)")


def create_huggingface_client():
    """Create and return a HuggingFace chat model client."""
    # Initialize LangChain HuggingFace client
    llm = HuggingFaceEndpoint(
        repo_id=HF_MODEL,
        huggingfacehub_api_token=HF_TOKEN,
        temperature=0.7,
        max_new_tokens=512,
        do_sample=True,
    )

    # Initialize chat model for better conversation handling
    chat_model = ChatHuggingFace(
        llm=llm,
        verbose=True
    )
    
    return chat_model

