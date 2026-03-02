import argparse
import os
import pandas as pd
from sentence_transformers import SentenceTransformer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    df = pd.read_parquet(os.path.join(args.data, "data.parquet"))

    model = SentenceTransformer("all-MiniLM-L6-v2")

    embeddings = model.encode(df["reviewText"].tolist(), show_progress_bar=True)

    emb_df = pd.DataFrame(embeddings)
    emb_df["asin"] = df["asin"].values
    emb_df["reviewerID"] = df["reviewerID"].values

    os.makedirs(args.out, exist_ok=True)
    emb_df.to_parquet(os.path.join(args.out, "data.parquet"))

    print("Semantic embeddings created.")


if __name__ == "__main__":
    main()