import os
from dotenv import load_dotenv
from pathlib import Path
from huggingface_hub import whoami

# Charger les variables depuis .env à la racine
ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    print("❌ HF_TOKEN n'est pas défini")
else:
    print("✅ HF_TOKEN trouvé :", HF_TOKEN[:10] + "..." + HF_TOKEN[-5:])
    try:
        info = whoami(token=HF_TOKEN)
        print("🎉 Authentification réussie !")
        print("Utilisateur :", info.get("name"))
        print("Plan :", info.get("orgs", "aucune organisation"))
    except Exception as e:
        print("⚠️ Erreur d'authentification :", e)
