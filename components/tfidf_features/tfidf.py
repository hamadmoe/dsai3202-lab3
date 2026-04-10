import argparse
import os
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--max_features", type=int, default=5000)
    parser.add_argument("--mode", type=str, required=True)  # train / inference
    parser.add_argument("--vectorizer", type=str)  # used ONLY for inference
    return parser.parse_args()


def main():
    args = parse_args()

    df = pd.read_parquet(os.path.join(args.data, "data.parquet"))
    text_column = "reviewText"

    os.makedirs(args.out, exist_ok=True)

    # ---------------- TRAIN MODE ----------------
    if args.mode == "train":
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words="english",
            ngram_range=(1, 1),
            min_df=5
        )

        X = vectorizer.fit_transform(df[text_column])
        vectorizer_path = os.path.join(args.out, "vectorizer.joblib")
        joblib.dump(vectorizer, vectorizer_path)

    # ---------------- INFERENCE MODE ----------------
    elif args.mode == "inference":
        if not args.vectorizer:
            raise ValueError("Vectorizer input path required for inference mode.")

        vectorizer_path = os.path.join(args.vectorizer, "vectorizer.joblib")
        vectorizer = joblib.load(vectorizer_path)
        X = vectorizer.transform(df[text_column])

    else:
        raise ValueError("Mode must be 'train' or 'inference'")

    # ---------------- CREATE DATAFRAME ----------------
    tfidf_df = pd.DataFrame(
        X.toarray(),
        columns=vectorizer.get_feature_names_out()
    )

    # Add entity keys
    tfidf_df["asin"] = df["asin"].values
    tfidf_df["reviewerID"] = df["reviewerID"].values

    # ---------------- SAVE FEATURES ----------------
    tfidf_df.to_parquet(os.path.join(args.out, "data.parquet"))

    print("TF-IDF features created.")
    print("Shape:", tfidf_df.shape)
    print("DEBUG: NEW TFIDF CODE RUNNING")
    print("TF-IDF shape:", X.shape)


if __name__ == "__main__":
    main()