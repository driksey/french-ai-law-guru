from faq_chatbot.utils import top_k_similar, build_embeddings_if_needed, load_faqs

def test_top_k_similar_tmp(tmp_path):
    faqs = [
        {"question":"How to pay?","answer":"By card"},
        {"question":"Refund policy","answer":"30 days"}
    ]
    p = tmp_path / "faqs.json"
    p.write_text(str(faqs))
    embeddings = build_embeddings_if_needed(faqs, emb_path=str(tmp_path/"emb.npy"))
    res = top_k_similar("payment", faqs, embeddings, k=1)
    assert len(res) == 1
