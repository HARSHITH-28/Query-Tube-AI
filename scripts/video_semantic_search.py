import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def search_videos(query, model, video_embeddings, df, top_k=5, threshold=0.3):
    
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, video_embeddings)[0]
    sorted_indices = np.argsort(similarities)[::-1]

    results = []
    for idx in sorted_indices[:top_k]:
        score = similarities[idx]
        if score >= threshold:
            video_id = df.iloc[idx]["video_id"]
            title = df.iloc[idx]["title"]
            youtube_link = f"https://www.youtube.com/watch?v={video_id}"
            results.append({
                "title": title,
                "video_id": video_id,
                "score": float(score),
                "link": youtube_link
            })
    return results

def display_results(results):
    if len(results) == 0:
        print("\nNo relevant videos found.")
        return

    print("\nTop Video Results:\n")
    for i, r in enumerate(results, start=1):
        print(f"{i}. {r['title']}")
        print(f"   Video ID: {r['video_id']}")
        print(f"   Similarity Score: {r['score']:.3f}")
        print(f"   YouTube Link: {r['link']}")
        print()

if __name__ == "__main__":
    print("\nAI Semantic Video Search\n")
    
    df = pd.read_parquet("data/video_index.parquet")
    embedding_cols = [c for c in df.columns if c.startswith("emb_")]
    video_embeddings = df[embedding_cols].values
    model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")
    
    while True:
        query = input("Enter your search query (or type 'exit'): ")
        if query.lower() == "exit":
            break
        results = search_videos(query, model, video_embeddings, df)
        display_results(results)