# src/embeddings.py
from sentence_transformers import SentenceTransformer
from pdf_processor import process_pdfs
import numpy as np
import pickle
import os


def create_embeddings(chunks, model_name="all-MiniLM-L6-v2", save_path="models/embeddings"):
    model = SentenceTransformer(model_name)
    texts = [chunk.page_content for chunk in chunks]
    embeddings = model.encode(texts)

    # Save embeddings and metadata
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(f"{save_path}/embeddings.pkl", "wb") as f:
        pickle.dump(embeddings, f)

    with open(f"{save_path}/texts.pkl", "wb") as f:
        pickle.dump(texts, f)

    model.save(f"{save_path}/model")
    return embeddings, texts, model


def load_embeddings(load_path="models/embeddings"):
    with open(f"{load_path}/embeddings.pkl", "rb") as f:
        embeddings = pickle.load(f)
    with open(f"{load_path}/texts.pkl", "rb") as f:
        texts = pickle.load(f)
    model = SentenceTransformer(f"{load_path}/model")
    return embeddings, texts, model


def find_similar_texts(query, embeddings, texts, model, top_k=3):
    query_embedding = model.encode(query)
    similarities = np.dot(embeddings, query_embedding)
    most_similar = np.argsort(similarities)[-top_k:][::-1]
    return [texts[i] for i in most_similar]