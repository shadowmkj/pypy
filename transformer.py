from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")


def get_embeddings(content):
    return model.encode(content, batch_size=32, show_progress_bar=False, device="mps")


if __name__ == "__main__":
    sentences = ["Llama 3.2 is great for chat", "Embeddings are useful for search"]
    embeddings = model.encode(sentences)
    print(embeddings)  # (2, 384)
    print(embeddings.shape)  # (2, 384)
