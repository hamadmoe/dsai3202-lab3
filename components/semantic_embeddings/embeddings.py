import argparse
import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    # ---------------- LOAD DATA ----------------
    df = pd.read_parquet(os.path.join(args.data, "data.parquet"))

    if "reviewText" not in df.columns:
        raise ValueError("reviewText column not found.")

    texts = df["reviewText"].astype(str).tolist()

    print("Loaded data:", len(texts))

    # ---------------- LOAD MODEL ----------------
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # ---------------- ENCODE (BATCHED + MEMORY SAFE) ----------------
    embeddings = model.encode(
        texts,
        batch_size=16,              # 🔥 prevents memory crash
        show_progress_bar=True,
        convert_to_numpy=True
    )

    print("Original embedding shape:", embeddings.shape)

    # ---------------- REDUCE DIMENSIONS ----------------
    # Keep only first 50 dimensions (VERY IMPORTANT)
    embeddings = embeddings[:, :50]

    print("Reduced embedding shape:", embeddings.shape)

    # Convert to float32 to reduce memory
    embeddings = embeddings.astype(np.float32)

    # ---------------- CREATE DATAFRAME ----------------
    emb_df = pd.DataFrame(
        embeddings,
        columns=[f"emb_{i}" for i in range(embeddings.shape[1])]
    )

    # Add keys
    emb_df["asin"] = df["asin"].values
    emb_df["reviewerID"] = df["reviewerID"].values

    print("Final embeddings DF shape:", emb_df.shape)

    # ---------------- SAVE ----------------
    os.makedirs(args.out, exist_ok=True)
    emb_df.to_parquet(os.path.join(args.out, "data.parquet"))

    print("Semantic embeddings created successfully.")


if __name__ == "__main__":
    main()