"""Encodage des textes en vecteurs denses (SBERT)."""
from functools import lru_cache
from typing import Protocol

import numpy as np

from src.config import EMBEDDING_MODEL


class Embedder(Protocol):
    """Tout objet exposant encode(list[str]) -> np.ndarray (n, d)."""

    def encode(self, texts: list[str]) -> np.ndarray: ...


class SbertEmbedder:
    """Wrapper autour de sentence-transformers, embeddings normalisés L2.

    Les textes du référentiel (300 parfums × 3 critères) sont identiques à
    chaque requête : on garde leurs embeddings en cache pour ne les calculer
    qu'une fois. Seules les phrases de l'utilisateur sont encodées à chaque fois.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        from sentence_transformers import SentenceTransformer  # import tardif

        self.model = SentenceTransformer(model_name)
        self._cache: dict[tuple[str, ...], np.ndarray] = {}

    def encode(self, texts: list[str]) -> np.ndarray:
        key = tuple(texts)
        if key not in self._cache:
            emb = self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True,
                                    batch_size=64, show_progress_bar=False)
            if len(texts) > 20:          # ne met en cache que les gros lots (référentiel)
                self._cache[key] = emb
            return emb
        return self._cache[key]


@lru_cache(maxsize=2)
def get_embedder(model_name: str = EMBEDDING_MODEL) -> SbertEmbedder:
    """Charge le modèle une seule fois par processus."""
    return SbertEmbedder(model_name)


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Similarité cosinus entre chaque ligne de a et chaque ligne de b."""
    a = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)
    b = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-12, None)
    return a @ b.T
