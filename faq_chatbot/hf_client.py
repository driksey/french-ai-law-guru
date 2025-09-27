# hf_client.py
import os
from dotenv import load_dotenv
from pathlib import Path
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.schema import HumanMessage, SystemMessage

# Charger le .env depuis la racine
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL = os.getenv("HF_MODEL", "google/gemma-2-2b-it")

if not HF_TOKEN:
    raise EnvironmentError("Set HF_TOKEN in environment (see .env.example)")

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


def generate_from_model(messages: list) -> str:
    """
    Génère une réponse depuis le modèle Hugging Face via LangChain.
    """
    try:
        # Convert messages to LangChain format
        langchain_messages = []
        for msg in messages:
            if msg["role"] == "system":
                langchain_messages.append(SystemMessage(content=msg["content"]))
            elif msg["role"] == "user":
                langchain_messages.append(HumanMessage(content=msg["content"]))
        
        # Generate response using LangChain
        response = chat_model.invoke(langchain_messages)
        return response.content
    except Exception as e:
        return f"Erreur lors de la génération: {str(e)}"
