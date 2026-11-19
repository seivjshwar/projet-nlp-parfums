"""Prétraitement des textes utilisateur et du référentiel."""
import re
import unicodedata


def normalize(text: str) -> str:
    """Normalise un texte : unicode NFC, espaces multiples, espaces en bord.

    On garde la casse et la ponctuation : les modèles SBERT les exploitent.
    """
    text = unicodedata.normalize("NFC", text or "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    """Découpe une réponse libre en phrases.

    Comparer phrase par phrase évite qu'une longue réponse « dilue » une
    compétence précise dans un seul embedding moyen.
    """
    text = normalize(text)
    if not text:
        return []
    parts = re.split(r"(?<=[.!?;])\s+|\n+", text)
    return [p.strip() for p in parts if len(p.strip()) > 2]


def prepare_user_texts(answers: list[str]) -> list[str]:
    """Transforme la liste des réponses libres en liste de phrases à encoder."""
    sentences: list[str] = []
    for answer in answers:
        sentences.extend(split_sentences(answer))
    return sentences
