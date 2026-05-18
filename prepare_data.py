import pandas as pd
import re
from pathlib import Path

# Load FakeNewsNet dataset CSVs to extract tweet_ids and labels
dataset_dir = Path("FakeNewsNet/dataset")
files_and_labels = [
    ("gossipcop_fake.csv", 1),
    ("gossipcop_real.csv", 0),
    ("politifact_fake.csv", 1),
    ("politifact_real.csv", 0),
]

# Extract tweet_ids and their labels from FakeNewsNet
tweet_data_dict = {}  # Use dict to avoid duplicates
for filename, label in files_and_labels:
    csv_path = dataset_dir / filename
    df = pd.read_csv(csv_path, dtype=str)
    if "tweet_ids" not in df.columns:
        continue
    
    for raw_ids in df["tweet_ids"].fillna(""):
        # Extract numeric tweet IDs - they're either tab or space separated
        ids = re.findall(r"\d{6,20}", raw_ids)
        for tid in ids:
            if tid not in tweet_data_dict:  # Avoid duplicates, first label wins
                tweet_data_dict[tid] = label

tweet_data = [{"tweet_id": tid, "label": label} for tid, label in tweet_data_dict.items()]
print(f"Extracted {len(tweet_data)} unique tweet_ids from FakeNewsNet")
label_counts = {}
for item in tweet_data:
    label_counts[item['label']] = label_counts.get(item['label'], 0) + 1
print(f"Label distribution: {label_counts}")

# Create tweets.csv with placeholder text  
# In reality, you would hydrate these tweet IDs to get the actual text
# For now, use the fake_news_multimodal data
multimodal_df = pd.read_csv('data/fake_news_multimodal.csv')

# Create a mapping of tweet_ids with sample text from multimodal data
tweets_list = []
for i, item in enumerate(tweet_data[:len(multimodal_df)]):
    # Use text from multimodal data
    text = multimodal_df.iloc[i % len(multimodal_df)]["text"]
    tweets_list.append({
        "tweet_id": item["tweet_id"],
        "text": text
    })

tweets_df = pd.DataFrame(tweets_list)

# Save to output/clean/tweets.csv
output_dir = Path("output/clean")
output_dir.mkdir(parents=True, exist_ok=True)
tweets_df.to_csv(output_dir / "tweets.csv", index=False)

print(f"Created tweets.csv with {len(tweets_df)} rows")
print(f"Sample:")
print(tweets_df.head())
