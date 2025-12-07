from sentence_transformers import SentenceTransformer

# Load same model you used for node embeddings
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_user_query(text: str):
    vector = model.encode(text).tolist()  # convert to Python list (Neo4j compatible)
    return vector

# # Example user query
# user_query = "Who are the top forwards in the 2023 season?"
# embedding = embed_user_query(user_query)

# print("User query embedding (vector length = {}):".format(len(embedding)))
# print(embedding[:10], "...")   # print first 10 dims
