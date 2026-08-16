import pandas as pd

# Load the datasets
fake = pd.read_csv("data/Fake.csv")
real = pd.read_csv("data/True.csv")

# Add labels
fake["label"] = 0
real["label"] = 1

# Combine fake and real news
df = pd.concat([fake, real], ignore_index=True)

# Remove rows with missing title or text
df = df.dropna(subset=["title", "text"])

# Combine title and article text
df["content"] = df["title"] + " " + df["text"]

# Keep only the columns needed for machine learning
df = df[["content", "label"]]

# Shuffle the dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save the prepared dataset
df.to_csv("data/news_dataset.csv", index=False)

# Display results
print("Dataset prepared successfully!")
print("Total articles:", len(df))

print("\nLabel distribution:")
print(df["label"].value_counts())

print("\nFirst 5 rows:")
print(df.head())