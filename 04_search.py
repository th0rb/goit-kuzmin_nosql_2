# scripts/04_search.py
import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

load_dotenv()

INDEX_NAME = "arxiv-papers"
MODEL_NAME = "allenai/specter2_base"
INPUT_PARQUET = "data/arxiv_subset.parquet"
INPUT_EMBEDDINGS = "data/arxiv_embeddings.npy"
NAMESPACE = "test"

TOP_K = 5

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(INDEX_NAME)
model = SentenceTransformer(MODEL_NAME)
df = pd.read_parquet(INPUT_PARQUET)  # для отримання повного abstract

# Запити
QUERY = "superconducting behavior"

print("Завантаження моделі SPECTER2...")
model = SentenceTransformer("allenai/specter2_base", device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu")

# ========================= ПІДКЛЮЧЕННЯ ДО PINECONE =========================

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    PINECONE_API_KEY = input("Введіть Pinecone API Key: ")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

print(f"Підключено до індексу: {INDEX_NAME}\n")

#  ФУНКЦІЯ КОДУВАННЯ ЗАПИТУ =========================

def encode_query(text: str):
    """Кодує запит у ембеддинг з нормалізацією"""
    embedding = model.encode(text, normalize_embeddings=True)
    return embedding.tolist()

#  1. ЧИСТИЙ СЕМАНТИЧНИЙ ПОШУК =========================

print(f"\n{'='*60}")
print(f"ЧИСТИЙ СЕМАНТИЧНИЙ ПОШУК")
print(f"Запит: {QUERY}")
print(f"{'='*60}")

query_embedding = encode_query(QUERY)

results = index.query(
    vector=query_embedding,
    top_k=TOP_K,
    namespace=NAMESPACE,
    include_metadata=True
)

print(results)

for i, match in enumerate(results['matches'], 1):
    meta = match['metadata']
    score = match['score']
    print(f"{i:2d}. Score: {score:.4f} | {meta.get('year', 'N/A')} | {meta.get('category', '')}")
    print(f"    {meta.get('title', 'No title')[:120]}...")
    print(f"    Abstract: {meta.get('abstract', '')[:180]}...\n")

# 2. ПОШУК З ФІЛЬТРАМИ =========================

print(f"\n{'='*60}")
print("ПОШУК З ФІЛЬТРАМИ")
print(f"{'='*60}")

# Приклад A: Reinforcement Learning, останні 5 років, cs.LG
print("Приклад A: Reinforcement Learning (cs.LG), після 2017 року")
filter_a = {
    "year": {"$gte": 2020},
    "category": {"$eq": "cs.LG"}
}

results_a = index.query(
    vector=query_embedding,
    top_k=5,
    filter=filter_a,
    namespace=NAMESPACE,
    include_metadata=True
)

for i, match in enumerate(results_a['matches'], 1):
    meta = match['metadata']
    print(f"{i}. {meta.get('title', '')[:100]}... | {meta.get('year')}")

# Приклад B: Старі статті (до 2015)
print("\nПриклад B: Статті до 2015 року")
filter_b = {"year": {"$lte": 2015}}

results_b = index.query(
    vector=query_embedding,
    top_k=5,
    filter=filter_b,
    namespace=NAMESPACE,
    include_metadata=True
)

for i, match in enumerate(results_b['matches'], 1):
    meta = match['metadata']
    print(f"{i}. {meta.get('title', '')[:100]}... | {meta.get('year')}")

# 3. ПОРІВНЯННЯ МЕТРИК НА ЛОКАЛЬНИХ ЕМБЕДДИНГАХ =========================

print(f"\n{'='*60}")
print("ПОРІВНЯННЯ МЕТРИК НА ЛОКАЛЬНИХ ЕМБЕДДИНГАХ")
print(f"{'='*60}")

# Завантаження даних
df = pd.read_parquet(INPUT_PARQUET)
embeddings = np.load(INPUT_EMBEDDINGS)

query_emb = model.encode(QUERY, normalize_embeddings=False)  # без нормалізації для порівняння

print(f"Запит: {QUERY}\n")

metrics = {
    "Cosine Similarity": lambda x: np.dot(x, query_emb) / (np.linalg.norm(x) * np.linalg.norm(query_emb) + 1e-8),
    "Dot Product": lambda x: np.dot(x, query_emb),
    "L2 Distance": lambda x: np.linalg.norm(x - query_emb)
}

for metric_name, metric_func in metrics.items():
    print(f"\n--- {metric_name} ---")
    scores = np.array([metric_func(emb) for emb in embeddings])
    
    if metric_name == "L2 Distance":
        top_indices = np.argsort(scores)[:TOP_K]      # найменша відстань
    else:
        top_indices = np.argsort(scores)[-TOP_K:][::-1]  # найбільша схожість
    
    for rank, idx in enumerate(top_indices, 1):
        row = df.iloc[idx]
        score = scores[idx]
        print(f"{rank}. Score: {score:.4f} | {row.get('year', 'N/A')} | {row.get('category', '')}")
        print(f"    {row.get('title', '')[:110]}...")
