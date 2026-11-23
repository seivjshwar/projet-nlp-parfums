"""Compare les 2 méthodes de similarité par famille sur des requêtes de test."""
import pandas as pd
from nlp_engine import (build_perfume_texts, load_perfumes, load_or_build_embeddings,
                        compute_perfume_similarities, compute_family_similarities)

df = build_perfume_texts(load_perfumes())
emb = load_or_build_embeddings(df)

requetes = {
    "frais, boisé, bureau": "Un parfum frais, élégant et légèrement boisé pour le bureau",
    "vanille, gourmand": "Un parfum sucré et gourmand à la vanille, réconfortant",
    "agrumes, citron": "Une odeur d'agrumes pétillante, comme un citron au soleil",
    "cuir, fumé": "Un parfum sombre de cuir et de fumée, très masculin",
    "fleurs, romantique": "Un bouquet de roses et de jasmin, doux et romantique",
}

for label, q in requetes.items():
    user = {"description_libre": q}
    sims = compute_perfume_similarities(user, df, emb)
    table = pd.DataFrame({
        "description": compute_family_similarities(user, df, emb, method="description"),
        "catalogue": compute_family_similarities(user, df, emb, perfume_sims=sims),
    })
    table["rang_desc"] = table["description"].rank(ascending=False).astype(int)
    table["rang_cata"] = table["catalogue"].rank(ascending=False).astype(int)
    print(f"\n######## {label} : « {q} »")
    print(table.sort_values("catalogue", ascending=False).round(3).to_string())
