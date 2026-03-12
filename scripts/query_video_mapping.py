import pandas as pd

df = pd.read_csv("data/cleaned_transcripts.csv")
queries = [
    "What is machine learning?",
    "What is reinforcement learning?",
    "How does gradient descent work?",
    "Explain neural networks",
    "What is artificial general intelligence?",
    "What is deep learning?",
    "How do transformers work?",
    "What is AI alignment?",
    "What is overfitting in machine learning?",
    "How do robots learn from data?"
]
mapping = []
for query in queries:
    keyword = query.lower().replace("what is", "").replace("how does", "").replace("explain", "").strip()
    matches = df[df["transcript"].str.contains(keyword, case=False, na=False)]
    if len(matches) > 0:
        video_id = matches.iloc[0]["video_id"]
        mapping.append((query, video_id))
    else:
        mapping.append((query, None))

mapping_df = pd.DataFrame(mapping, columns=["query", "relevant_video_id"])
mapping_df.to_csv("data/query_video.csv", index=False)
print(mapping_df)