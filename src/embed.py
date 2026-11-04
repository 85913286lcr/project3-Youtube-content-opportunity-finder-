"""Sentence embeddings for comments and video titles (all-MiniLM-L6-v2, 384 dims).

Embedding is a fixed pretrained transformation, nothing is learned from our data,
so running it on the whole dataset does not leak holdout information.

Usage:
    python -m src.embed
"""
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from src import config

COMMENT_EMB = config.DATA_PROCESSED / "comment_embeddings.npy"
TITLE_EMB = config.DATA_PROCESSED / "title_embeddings.npy"


def get_model():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    return SentenceTransformer(config.EMBED_MODEL, device=device)


def encode(texts, model=None, batch_size=128):
    model = model or get_model()
    return model.encode(list(texts), batch_size=batch_size, show_progress_bar=True,
                        normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)


def load_comment_embeddings():
    return np.load(COMMENT_EMB)


def load_title_embeddings():
    return np.load(TITLE_EMB)


def run():
    comments = pd.read_csv(config.DATA_PROCESSED / "comments_clean.csv")
    videos = pd.read_csv(config.DATA_PROCESSED / "videos_clean.csv")
    model = get_model()
    np.save(COMMENT_EMB, encode(comments["text_clean"], model))
    np.save(TITLE_EMB, encode(videos["title_clean"], model))
    print("saved", COMMENT_EMB.name, TITLE_EMB.name)


if __name__ == "__main__":
    run()
