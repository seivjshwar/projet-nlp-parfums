import numpy as np
import pandas as pd
from nlp_engine import (build_perfume_texts, load_perfumes,
                        load_or_build_embeddings, compute_perfume_similarities)

df = build_perfume_texts(load_perfumes())
emb = load_or_build_embeddings(df)
pd.set_option("display.width", 220)
pd.set_option("display.max_colwidth", 110)

# --- Diagnostic : les phrases les plus « passe-partout » (hubs)
sentences = [s for sents in df["description_sentences"] for s in sents]
hubs = pd.DataFrame({"parfum": df["Name"].to_numpy()[emb["desc_owner"]],
                     "popularite": emb["desc_hub"], "phrase": sentences})
print("=== 5 phrases les plus populaires (hubs) ===")
print(hubs.nlargest(5, "popularite").round(3).to_string(index=False))
print(f"Popularité : moyenne {hubs.popularite.mean():.3f}, min {hubs.popularite.min():.3f}, "
      f"max {hubs.popularite.max():.3f}")

requetes = [
    "Un parfum frais, élégant et légèrement boisé pour le bureau",
    "Un parfum sucré et gourmand à la vanille, réconfortant",
    "Une odeur d'agrumes pétillante, comme un citron au soleil",
]

for q in requetes:
    sims = compute_perfume_similarities({"description_libre": q}, df, emb)
    sims = sims.join(df[["Name", "Family"]])
    print(f"\n######## « {q} »")
    for col in ["sim_profile", "sim_description_raw", "sim_description"]:
        top = sims.nlargest(10, col)
        print(f"--- Top 10 {col} : familles = {top['Family'].value_counts().to_dict()}")
        print(top[["Name", "Family", col]].head(5).round(3).to_string(index=False))
    best = sims.loc[sims["sim_description"].idxmax()]
    print(f"--- Meilleure paire (corrigée) : « {best['best_perfume_phrase']} »")