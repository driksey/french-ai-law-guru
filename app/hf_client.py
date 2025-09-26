import os
from huggingface_hub import InferenceApi

HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL = os.getenv("HF_MODEL", "google/flan-t5-small")

if not HF_TOKEN:
    raise EnvironmentError("Set HF_TOKEN in environment (see .env.example)")

_client = InferenceApi(repo_id=HF_MODEL, token=HF_TOKEN)

def generate_from_model(prompt: str, max_new_tokens: int = 200) -> str:
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": max_new_tokens}}
    res = _client(payload)
    # handle multiple response shapes
    if isinstance(res, dict):
        return res.get("generated_text", "") or str(res)
    if isinstance(res, list) and res and isinstance(res[0], dict):
        return res[0].get("generated_text", "") or str(res)
    return str(res)