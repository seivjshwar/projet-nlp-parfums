import pandas as pd
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
tok = model.tokenizer   # l'outil qui découpe le texte en tokens

# 1. Charger le CSV de l'équipe
df = pd.read_csv("data/reference/perfumes_200_clean.csv")
print("Taille :", df.shape)
print("Colonnes :", df.columns.tolist())

# 2. Longueur en tokens de quelques colonnes texte
for col in ["Keywords", "OlfactoryNotes", "Occasion", "Description", "SearchText"]:
    n = df[col].fillna("").map(lambda t: len(tok.tokenize(t)))
    print(f"{col:15s} médiane {n.median():5.0f} tokens | max {n.max():4d} | "
          f"tronqués : {(n > model.max_seq_length - 2).sum()} / {len(df)}")

# 3. Ce que le modèle "voit" vraiment pour le 1er parfum
texte = df["SearchText"][0]
ids = tok(texte, truncation=True, max_length=model.max_seq_length)["input_ids"]
print("\n=== Ce que le modèle lit ===")
print(tok.decode(ids, skip_special_tokens=True))
print("\n=== Texte complet ===")
print(texte)