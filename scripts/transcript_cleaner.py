import pandas as pd
import re

def load_dataset():
    df = pd.read_csv("data/enriched_dataset.csv")
    required_columns = [
        "video_id",
        "title",
        "published_date",
        "transcript"
    ]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    print("Dataset loaded successfully")
    print("Dataset shape:", df.shape)
    return df

def clean_text(text):
    if pd.isna(text):
        return text
    text = str(text)
    text = re.sub(r'[\n\t]', ' ', text)
    text = re.sub(r'[#$@*]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def clean_dataset(df):
    print("Cleaning titles...")
    df["title"] = df["title"].apply(clean_text)
    print("Cleaning transcripts...")
    df["transcript"] = df["transcript"].apply(clean_text)
    return df

def handle_missing_transcripts(df):
    missing_before = df["transcript"].isna().sum()
    print("Missing transcripts before cleaning:", missing_before)
    df = df.dropna(subset=["transcript"])
    missing_after = df["transcript"].isna().sum()
    print("Missing transcripts after cleaning:", missing_after)
    print("Dataset shape after removing missing transcripts:", df.shape)
    return df
    
def normalize_dataset_format(df):
    print("Normalizing dataset format...")
    if "published_date" in df.columns:
        df = df.rename(columns={"published_date": "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df[[
        "video_id",
        "title",
        "datetime",
        "transcript"
    ]]
    print("Dataset normalized.")
    print("Columns:", df.columns.tolist())
    return df

if __name__ == "__main__":
    df = load_dataset()
    df = clean_dataset(df)
    df = handle_missing_transcripts(df)
    df = normalize_dataset_format(df)
    df.to_csv("data/cleaned_transcripts.csv", index=False)
    print("Cleaned dataset saved to:")
    print("data/cleaned_transcripts.csv")