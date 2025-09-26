# hf_client.py
import os
from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path

# Charger le .env depuis la racine
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL = os.getenv("HF_MODEL", "google/gemma-2-2b-it:nebius")

if not HF_TOKEN:
    raise EnvironmentError("Set HF_TOKEN in environment (see .env.example)")

# Client OpenAI vers l'API Hugging Face
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_TOKEN
)


def generate_from_model(messages: list) -> str:
    """
    Génère une réponse depuis le modèle Hugging Face via le client OpenAI.
    """
    try:
        completion = client.chat.completions.create(
            model=HF_MODEL,
            messages=messages,
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Erreur lors de la génération: {str(e)}"
