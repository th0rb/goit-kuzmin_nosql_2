# scripts/02_embed.py

import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from huggingface_hub import login

# Load environment variables from the .env file
load_dotenv()

hf_Token = os.getenv("HF_TOKEN")
login(token=hf_Token)

MODEL_NAME = "allenai/specter2_base"
BATCH_SIZE = 64
DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
OUTPUT_PATH = "data/arxiv_embeddings.npy"

df = pd.read_parquet("data/arxiv_subset.parquet")  # для отримання повного abstract

print(df.head)

# Об'єднуємо title і abstract
df['text_for_embedding'] = df['title'].fillna('') + " [SEP] " + df['abstract'].fillna('')

texts = df['text_for_embedding'].tolist()

# ========================= ЗАВАНТАЖЕННЯ МОДЕЛІ =========================

print(f"Завантаження моделі {MODEL_NAME}...")
model = SentenceTransformer(MODEL_NAME, device=DEVICE)

# ========================= ГЕНЕРАЦІЯ ЕМБЕДДИНГІВ =========================

print("Генерація ембеддингів...")

embeddings = model.encode(
    texts,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    normalize_embeddings=True,      # L2-нормалізація
    convert_to_numpy=True
)

print(f"\nЕмбеддинги згенеровано! Розмір: {embeddings.shape}")

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

np.save(OUTPUT_PATH, embeddings)
print(f"Ембеддинги збережено у: {OUTPUT_PATH}")
print(f"Норма першого ембеддингу: {np.linalg.norm(embeddings[0])}")


# Додатково зберігаємо індекси для зручності
df[['title', 'abstract']].to_parquet("data/arxiv_metadata.parquet", index=False)
print("Метадані збережено у data/arxiv_metadata.parquet")

print("\nГотово!")