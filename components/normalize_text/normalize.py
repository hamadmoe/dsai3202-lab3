import argparse
import os
import re
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    return parser.parse_args()


def normalize_text(text):
    if pd.isna(text):
        return ""

    text = str(text)
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " <url> ", text)
    text = re.sub(r"\d+", " <number> ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    return text


def main():
    args = parse_args()

    # Load dataset
    input_path = os.path.join(args.data, "data.parquet")
    df = pd.read_parquet(input_path)

    # ---------------- FIX: DETECT TEXT COLUMN ----------------
    if "reviewText" in df.columns:
        text_column = "reviewText"
    elif "reviewText_x" in df.columns:
        text_column = "reviewText_x"
    elif "reviewText_y" in df.columns:
        text_column = "reviewText_y"
    else:
        raise ValueError(
            "No review text column found (expected reviewText / reviewText_x / reviewText_y)"
        )

    print(f"Using text column: {text_column}")

    # ---------------- NORMALIZE ----------------
    df[text_column] = df[text_column].apply(normalize_text)

    # ---------------- FILTER SHORT REVIEWS ----------------
    df = df[df[text_column].str.len() >= 10]

    # ---------------- OPTIONAL: STANDARDIZE COLUMN NAME ----------------
    # Rename to 'reviewText' so downstream steps are consistent
    if text_column != "reviewText":
        df = df.rename(columns={text_column: "reviewText"})

    # ---------------- SAVE ----------------
    os.makedirs(args.out, exist_ok=True)
    output_path = os.path.join(args.out, "data.parquet")
    df.to_parquet(output_path)

    print("Rows after normalization:", len(df))
    print("Columns:", df.columns.tolist())


if __name__ == "__main__":
    main()