import argparse
import os
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=str, required=True)
    parser.add_argument("--sentiment", type=str, required=True)
    parser.add_argument("--tfidf", type=str, required=True)
    parser.add_argument("--embeddings", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    length_df = pd.read_parquet(os.path.join(args.length, "data.parquet"))
    sentiment_df = pd.read_parquet(os.path.join(args.sentiment, "data.parquet"))
    tfidf_df = pd.read_parquet(os.path.join(args.tfidf, "data.parquet"))
    emb_df = pd.read_parquet(os.path.join(args.embeddings, "data.parquet"))

    # Merge sequentially on entity keys
    df = length_df.merge(sentiment_df, on=["asin", "reviewerID"])
    df = df.merge(tfidf_df, on=["asin", "reviewerID"])
    df = df.merge(emb_df, on=["asin", "reviewerID"])

    os.makedirs(args.out, exist_ok=True)
    df.to_parquet(os.path.join(args.out, "data.parquet"))

    print("All features merged.")
    print("Final shape:", df.shape)


if __name__ == "__main__":
    main()