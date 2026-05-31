# scripts/03_load_to_pinecone.py
import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

INPUT_PARQUET = "data/arxiv_subset.parquet"
INPUT_EMBEDDINGS = "data/arxiv_embeddings.npy"
INDEX_NAME = "arxiv-papers"
NAMESPACE = "test"
VECTOR_DIM = 768
BATCH_SIZE = 200   # Pinecone рекомендує батчі до 200 векторів

# Ініціалізація клієнта
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

# Створюємо індекс (якщо не існує)
if INDEX_NAME not in pc.list_indexes().names():
    print(f"Створення індексу '{INDEX_NAME}'...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=VECTOR_DIM,
        metric="cosine", 
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    print("Індекс створено. Очікування 5 секунд...")
    time.sleep(5)
else:
    print(f"Індекс '{INDEX_NAME}' вже існує.")

index = pc.Index(INDEX_NAME)
print(f"Підключено до індексу: {INDEX_NAME}\n")

print("Завантаження даних...")
df = pd.read_parquet(INPUT_PARQUET)
embeddings = np.load(INPUT_EMBEDDINGS)

print(f"Завантажено {len(df)} записів і {embeddings.shape[0]} ембеддингів.")


print("Початок завантаження в Pinecone...")

def truncate_text(text, max_length):
    if pd.isna(text) or text is None:
        return ""
    text = str(text)
    return text[:max_length] if len(text) > max_length else text

batch = []
for i in tqdm(range(len(df)), desc="Завантаження векторів"):
    row = df.iloc[i]
    
    vector = embeddings[i].tolist()
    paper_id = f"paper_{i}"
    
    metadata = {
        "arxiv_id": str(row.get("id", f"unknown_{i}")),
        "title": truncate_text(row.get("title", ""), 500),
        "abstract": truncate_text(row.get("abstract", ""), 500),
        "authors": truncate_text(row.get("authors", ""), 200),
        "year": int(row.get("year", 0)) if pd.notna(row.get("year")) else 0,
        "category": str(row.get("category", ""))[:100],
    }
    
    batch.append({
        "id": paper_id,
        "values": vector,
        "metadata": metadata
    })
    
    # Завантажуємо батч
    if len(batch) >= BATCH_SIZE:
        index.upsert(vectors=batch, namespace=NAMESPACE)
        batch = []

# Завантажуємо останній батч
if batch:
    index.upsert(vectors=batch, namespace=NAMESPACE)

print("\nЗавантаження завершено!")

# ========================= ПІДСУМОК =========================

stats = index.describe_index_stats()
total_vectors = stats.get('total_vector_count', 0)

print(f"\n=== РЕЗУЛЬТАТ ===")
print(f"Індекс: {INDEX_NAME}")
print(f"Загальна кількість векторів: {total_vectors}")
print(f"Namespace: '{NAMESPACE}'")