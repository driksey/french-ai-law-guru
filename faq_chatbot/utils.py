import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

EMB_MODEL = "all-MiniLM-L6-v2"

def load_faqs(path="docs/faqs.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_embeddings_if_needed(faqs, emb_path="faq_chatbot/embeddings.npy"):
    if os.path.exists(emb_path):
        return np.load(emb_path)
    model = SentenceTransformer(EMB_MODEL)
    texts = [item["question"] + " " + item.get("answer", "") for item in faqs]
    emb = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    np.save(emb_path, emb)
    return emb

def top_k_similar(query, faqs, embeddings, k=3):
    model = SentenceTransformer(EMB_MODEL)
    q_emb = model.encode([query], convert_to_numpy=True)
    sims = cosine_similarity(q_emb, embeddings)[0]
    idx = sims.argsort()[::-1][:k]
    return [(int(i), float(sims[i]), faqs[i]) for i in idx]
