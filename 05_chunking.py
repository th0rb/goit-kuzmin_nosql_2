# scripts/05_chunking.py
import os
import re
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

load_dotenv()

MODEL_NAME = "allenai/specter2_base"
VECTOR_DIM = 768
NUM_ARTICLES = 30
CHUNK_SIZE = 180          # слів для fixed chunking
CHUNK_OVERLAP = 40        # слів перекриття
MAX_WORDS_SEMANTIC = 200  # для semantic chunking

INDEX_FIXED = "arxiv-chunks-fixed"
INDEX_SEMANTIC = "arxiv-chunks-semantic"

TEST_QUERIES = [
    "recognize objects in pictures",
    "reinforcement learning",
    "large models limitations",
    "quantum computing algorithms"
]

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
model = SentenceTransformer(MODEL_NAME)
df = pd.read_parquet("data/arxiv_subset.parquet")


# Вибираємо 30 статей з найдовшими анотаціями
df['abstract_length'] = df['abstract'].fillna('').str.split().str.len()
df_selected = df.nlargest(NUM_ARTICLES, 'abstract_length').copy().reset_index(drop=True)

print(f"Вибрано {len(df_selected)} статей з найдовшими анотаціями.\n")

# ========================= МОДЕЛЬ =========================

print("Завантаження моделі SPECTER2...")
model = SentenceTransformer("allenai/specter2_base", device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu")

def create_index(index_name):
    if index_name not in pc.list_indexes().names():
        print(f"Створення індексу: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=VECTOR_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        time.sleep(5)
    return pc.Index(index_name)

index_fixed = create_index(INDEX_FIXED)
index_semantic = create_index(INDEX_SEMANTIC)

# ========================= ФУНКЦІЇ ЧАНКІНГУ =========================

def fixed_size_chunking(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def semantic_chunking(text, max_words=MAX_WORDS_SEMANTIC):
    # Проста семантична розбивка по реченнях
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = []
    current_words = 0

    for sentence in sentences:
        sentence_words = len(sentence.split())
        if current_words + sentence_words > max_words and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_words = sentence_words
        else:
            current_chunk.append(sentence)
            current_words += sentence_words
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks


def process_and_upload(chunks_func, index, index_name, chunk_type):
    print(f"\n=== Обробка {chunk_type} чанків ===")
    vectors = []
    batch_size = 100
    total_chunks = 0

    for idx, row in tqdm(df_selected.iterrows(), total=len(df_selected), desc=f"Чанкування {chunk_type}"):
        abstract = str(row.get('abstract', ''))
        title = str(row.get('title', ''))
        arxiv_id = str(row.get('id', f'unknown_{idx}'))
        
        chunks = chunks_func(abstract)
        total_chunks += len(chunks)

        for chunk_num, chunk_text in enumerate(chunks):
            # Генерація ембеддингу
            embedding = model.encode(
                title + " [SEP] " + chunk_text, 
                normalize_embeddings=True
            ).tolist()

            vector_id = f"{arxiv_id}_chunk_{chunk_num}_{chunk_type}"

            metadata = {
                "arxiv_id": arxiv_id,
                "title": title[:300],
                "chunk_text": chunk_text[:1000],
                "chunk_number": chunk_num,
                "year": int(row.get('year', 0)),
                "category": str(row.get('category', '')),
                "chunk_type": chunk_type
            }

            vectors.append({
                "id": vector_id,
                "values": embedding,
                "metadata": metadata
            })

            # Завантаження батчами
            if len(vectors) >= batch_size:
                index.upsert(vectors=vectors)
                vectors = []

    # Останній батч
    if vectors:
        index.upsert(vectors=vectors)

    print(f"Завантажено {total_chunks} {chunk_type} чанків у індекс {index_name}")

# ========================= ЗАПУСК ЧАНКІНГУ =========================

process_and_upload(fixed_size_chunking, index_fixed, INDEX_FIXED, "fixed")
process_and_upload(semantic_chunking, index_semantic, INDEX_SEMANTIC, "semantic")

# ========================= ПОШУК ПО ЧАНКАХ =========================

def search_in_index(index, query, top_k=5):
    query_emb = model.encode(query, normalize_embeddings=True).tolist()
    return index.query(vector=query_emb, top_k=top_k, include_metadata=True)

print("\n" + "="*70)
print("ТЕСТУВАННЯ ПОШУКУ ПО ЧАНКАХ")
print("="*70)

for query in TEST_QUERIES:
    print(f"\nЗапит: '{query}'")
    print("-" * 50)
    
    # Fixed chunks
    res_fixed = search_in_index(index_fixed, query)
    print("Fixed-size chunks:")
    for i, match in enumerate(res_fixed['matches'], 1):
        m = match['metadata']
        print(f"{i}. [{match['score']:.4f}] {m['title'][:90]}...")
        print(f"    Чанк {m['chunk_number']}: {m['chunk_text'][:180]}...\n")
    
    # Semantic chunks
    res_sem = search_in_index(index_semantic, query)
    print("Semantic chunks:")
    for i, match in enumerate(res_sem['matches'], 1):
        m = match['metadata']
        print(f"{i}. [{match['score']:.4f}] {m['title'][:90]}...")
        print(f"    Чанк {m['chunk_number']}: {m['chunk_text'][:180]}...\n")

print("Скрипт завершено успішно!")