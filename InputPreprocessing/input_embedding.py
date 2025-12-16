from sentence_transformers import SentenceTransformer

MODEL_MINILM = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_MPNET = "sentence-transformers/all-mpnet-base-v2"

models = {
    "minilm": SentenceTransformer(MODEL_MINILM),
    "mpnet": SentenceTransformer(MODEL_MPNET)
}


def embed_user_query(text: str, model_choice: str = "minilm"):
    """
    Convert user text query into an embedding using the SELECTED model.
    """
    model = models[model_choice]
    vector = model.encode(text).tolist()
    return vector