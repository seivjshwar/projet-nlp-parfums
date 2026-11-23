from sentence_transformers import SentenceTransformer, util

# 1. Charger le modèle (téléchargé la première fois, ~470 Mo, puis mis en cache)
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# 2. Quelques phrases : 2 proches en sens (FR / EN), 1 sans rapport
phrases = [
    "Un parfum frais et léger pour l'été",
    "A light and fresh summer fragrance",
    "Un parfum sucré à la vanille pour le soir",
]

# 3. Encoder = transformer chaque phrase en vecteur
emb = model.encode(phrases, normalize_embeddings=True)
print("Forme de la matrice :", emb.shape)      # (nb de phrases, dimension)
print("Début du 1er vecteur :", emb[0][:5])

# 4. Similarité cosinus de chaque phrase avec chaque autre
print(util.cos_sim(emb, emb))

# 5. Combien de mots (tokens) le modèle accepte-t-il au maximum ?
print("Longueur max :", model.max_seq_length, "tokens")