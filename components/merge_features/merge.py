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

    # ---------------- LOAD ----------------
    length_df = pd.read_parquet(os.path.join(args.length, "data.parquet"))
    sentiment_df = pd.read_parquet(os.path.join(args.sentiment, "data.parquet"))
    tfidf_df = pd.read_parquet(os.path.join(args.tfidf, "data.parquet"))
    emb_df = pd.read_parquet(os.path.join(args.embeddings, "data.parquet"))

    print("Loaded shapes:")
    print("Length:", length_df.shape)
    print("Sentiment:", sentiment_df.shape)
    print("TF-IDF:", tfidf_df.shape)
    print("Embeddings:", emb_df.shape)

    # ---------------- HANDLE TF-IDF ----------------
    tfidf_feature_cols = [col for col in tfidf_df.columns if col not in ["asin", "reviewerID"]]
    print("Original TF-IDF cols:", len(tfidf_feature_cols))

    # Drop TF-IDF for large datasets (train)
    if len(tfidf_df) > 30000:
        print("Large dataset detected → dropping TF-IDF for train")
        use_tfidf = False
        tfidf_df = None
    else:
        use_tfidf = True

        # limit features
        tfidf_feature_cols = tfidf_feature_cols[:150]
        tfidf_df = tfidf_df[["asin", "reviewerID"] + tfidf_feature_cols]

        # reduce memory
        tfidf_df[tfidf_feature_cols] = tfidf_df[tfidf_feature_cols].astype("float32")

        print("Using TF-IDF:", tfidf_df.shape)

    # ---------------- CLEAN DATAFRAMES ----------------

    # ---- LENGTH (keep only needed columns)
    base_cols = ["asin", "reviewerID"]

    if "overall" in length_df.columns:
        base_cols.append("overall")
    elif "overall_x" in length_df.columns:
        length_df["overall"] = length_df["overall_x"]
        base_cols.append("overall")
    else:
        raise ValueError("No 'overall' column found.")

    length_feature_cols = [col for col in length_df.columns if "review_length" in col]
    length_df = length_df[base_cols + length_feature_cols]

    # ---- SENTIMENT
    sentiment_cols = [
        "asin", "reviewerID",
        "sentiment_pos", "sentiment_neg",
        "sentiment_neu", "sentiment_compound"
    ]
    sentiment_df = sentiment_df[[col for col in sentiment_cols if col in sentiment_df.columns]]

    # ---- EMBEDDINGS
    emb_cols = [col for col in emb_df.columns if col not in ["asin", "reviewerID"]]
    emb_df = emb_df[["asin", "reviewerID"] + emb_cols]

    print("After cleaning:")
    print("Length:", length_df.shape)
    print("Sentiment:", sentiment_df.shape)
    print("Embeddings:", emb_df.shape)

    # ---------------- MERGE ----------------
    df = length_df.merge(sentiment_df, on=["asin", "reviewerID"], how="inner")
    del sentiment_df

    if use_tfidf:
        df = df.merge(tfidf_df, on=["asin", "reviewerID"], how="inner")
        del tfidf_df

    df = df.merge(emb_df, on=["asin", "reviewerID"], how="inner")
    del emb_df

    # ---------------- FINAL CHECK ----------------
    required_cols = ["asin", "reviewerID", "overall"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # ---------------- SAVE ----------------
    os.makedirs(args.out, exist_ok=True)
    df.to_parquet(os.path.join(args.out, "data.parquet"))

    print("All features merged successfully.")
    print("Final shape:", df.shape)
    print("Total columns:", len(df.columns))


if __name__ == "__main__":
    main()