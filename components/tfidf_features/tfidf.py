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
    parser.add_argument("--mode", type=str, required=True)  # train / infer
    parser.add_argument("--vectorizer", type=str)
    return parser.parse_args()


def main():
    args = parse_args()

    df = pd.read_parquet(os.path.join(args.data, "data.parquet"))
    text_column = "reviewText"

    os.makedirs(args.out, exist_ok=True)

    if args.mode == "train":
        vectorizer = TfidfVectorizer(
            max_features=args.max_features,
            stop_words="english",
            ngram_range=(1, 2)
        )

        X = vectorizer.fit_transform(df[text_column])

        joblib.dump(vectorizer, args.vectorizer)

    else:
        vectorizer = joblib.load(args.vectorizer_path)
        X = vectorizer.transform(df[text_column])

    tfidf_df = pd.DataFrame(
        X.toarray(),
        columns=vectorizer.get_feature_names_out()
    )

    # Add entity keys
    tfidf_df["asin"] = df["asin"].values
    tfidf_df["reviewerID"] = df["reviewerID"].values

    tfidf_df.to_parquet(os.path.join(args.out, "data.parquet"))

    print("TF-IDF features created.")


if __name__ == "__main__":
    main()