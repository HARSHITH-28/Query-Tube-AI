import pandas as pd

df = pd.read_csv("data/enriched_dataset.csv")
print("Total rows:", len(df))
non_empty = df["transcript"].notna().sum()
print("Rows with transcripts:", non_empty)
missing = df["transcript"].isna().sum()
print("Rows without transcripts:", missing)
print("Transcript coverage:", round((non_empty / len(df)) * 100, 2), "%")
transcripts_df = df[["video_id", "title", "transcript"]].dropna()
print("\nSample transcripts:\n")
for idx, row in transcripts_df.head(3).iterrows():
    print(f"Video ID: {row['video_id']}")
    print(f"Title: {row['title']}")
    print(f"Transcript (first 200 chars): {row['transcript'][:200]}...\n")
transcript_lengths = transcripts_df["transcript"].apply(len)
print(f"Longest transcript: {transcript_lengths.max()}")
print(f"Shortest transcript: {transcript_lengths.min()}")
print(f"Average transcript length: {transcript_lengths.mean():.2f}")