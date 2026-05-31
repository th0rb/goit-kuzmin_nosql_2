# scripts/06_hybrid_search.py
import os
import math
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

load_dotenv()

INDEX_NAME = "arxiv-papers"
MODEL_NAME = "allenai/specter2_base"
TOP_K = 10   # беремо ширше, щоб RRF міг переранжувати
RRF_K = 60   # кількість топ-документів для RRF

TEST_QUERIES = [
    "BERT fine-tuning",
    "Yann LeCun convolutional networks",
    "making computers understand human emotions from text"
]

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(INDEX_NAME)
model = SentenceTransformer(MODEL_NAME)
df = pd.read_parquet("data/arxiv_subset.parquet").reset_index(drop=True)

# Об'єднуємо title + abstract для BM25
df['text_for_bm25'] = df['title'].fillna('') + " " + df['abstract'].fillna('')
corpus = df['text_for_bm25'].tolist()

print(f"Корпус для BM25: {len(corpus)} документів\n")

# ========================= BM25 ІНДЕКС =========================

print("Побудова BM25 індексу...")
tokenized_corpus = [doc.lower().split() for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)

print("BM25 індекс побудовано.\n")

# ========================= МОДЕЛЬ ТА PINECONE =========================

print("Завантаження моделі SPECTER2...")
model = SentenceTransformer("allenai/specter2_base", device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu")

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY") or input("Введіть Pinecone API Key: "))
index = pc.Index(INDEX_NAME)

print(f"Підключено до Pinecone індексу: {INDEX_NAME}\n")

# ========================= ФУНКЦІЇ ПОШУКУ =========================

def bm25_search(query: str, top_k=TOP_K):
    """Пошук за допомогою BM25"""
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[-top_k:][::-1]
    
    results = []
    for rank, idx in enumerate(top_indices):
        row = df.iloc[idx]
        results.append({
            "id": f"paper_{idx}",
            "score": float(scores[idx]),
            "title": row['title'],
            "abstract": row['abstract'][:300],
            "year": int(row.get('year', 0)),
            "rank": rank + 1
        })
    return results


def vector_search(query: str, top_k=TOP_K):
    """Векторний пошук через Pinecone"""
    embedding = model.encode(query, normalize_embeddings=True).tolist()
    
    results = index.query(
        vector=embedding,
        top_k=top_k,
        include_metadata=True
    )
    
    output = []
    for match in results['matches']:
        meta = match['metadata']
        output.append({
            "id": match['id'],
            "score": match['score'],
            "title": meta.get('title', ''),
            "abstract": meta.get('abstract', '')[:300],
            "year": meta.get('year', 0),
            "rank": len(output) + 1
        })
    return output


def reciprocal_rank_fusion(results_list, k=RRF_K):
    """Reciprocal Rank Fusion (RRF)"""
    score_dict = {}
    
    for results in results_list:
        for rank, doc in enumerate(results, 1):
            doc_id = doc['id']
            if doc_id not in score_dict:
                score_dict[doc_id] = {
                    "title": doc['title'],
                    "abstract": doc['abstract'],
                    "year": doc['year'],
                    "rrf_score": 0.0
                }
            score_dict[doc_id]["rrf_score"] += 1.0 / (rank + k)
    
    # Сортуємо за RRF score
    fused = sorted(score_dict.items(), key=lambda x: x[1]["rrf_score"], reverse=True)
    
    return [
        {
            "id": doc_id,
            "rrf_score": info["rrf_score"],
            "title": info["title"],
            "abstract": info["abstract"],
            "year": info["year"],
            "rank": i + 1
        }
        for i, (doc_id, info) in enumerate(fused[:TOP_K])
    ]


# main code part #

print("="*80)
print("ГІБРИДНИЙ ПОШУК (BM25 + Vector + RRF)")
print("="*80)

for query in TEST_QUERIES:
    print(f"\n🔍 Запит: '{query}'")
    print("-" * 70)
    
    # 1. BM25
    bm25_results = bm25_search(query, top_k=TOP_K)
    print(f"{'BM25':<12} | Топ-5")
    for r in bm25_results[:5]:
        print(f"  {r['rank']:2d}. [{r['score']:.4f}] {r['title'][:95]}...")
    
    # 2. Vector Search
    vector_results = vector_search(query, top_k=TOP_K)
    print(f"\n{'Vector Search':<12} | Топ-5")
    for r in vector_results[:5]:
        print(f"  {r['rank']:2d}. [{r['score']:.4f}] {r['title'][:95]}...")
    
    # 3. Hybrid (RRF)
    hybrid_results = reciprocal_rank_fusion([bm25_results, vector_results])
    print(f"\n{'Hybrid RRF':<12} | Топ-5")
    for r in hybrid_results[:5]:
        print(f"  {r['rank']:2d}. [RRF={r['rrf_score']:.4f}] {r['title'][:95]}...")
    
    print("\n" + "─"*70)

print("\nСкрипт завершено успішно!")
