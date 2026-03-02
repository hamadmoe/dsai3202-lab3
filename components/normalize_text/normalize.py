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

    # Convert to string (safety)
    text = str(text)

    # Lowercase
    text = text.lower()

    # Replace URLs
    text = re.sub(r"http\S+|www\S+", " <url> ", text)

    # Replace numbers
    text = re.sub(r"\d+", " <number> ", text)

    # Remove punctuation (keep only letters, numbers and spaces)
    text = re.sub(r"[^\w\s]", " ", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)

    # Trim leading/trailing whitespace
    text = text.strip()

    return text


def main():
    args = parse_args()

    # Load dataset (expects parquet file inside folder)
    input_path = os.path.join(args.data, "data.parquet")
    df = pd.read_parquet(input_path)

    # Make sure column exists (adjust if your column name is different)
    text_column = "reviewText"

    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in dataset.")

    # Apply normalization
    df[text_column] = df[text_column].apply(normalize_text)

    # Remove empty or very short reviews (<10 characters)
    df = df[df[text_column].str.len() >= 10]

    # Create output directory
    os.makedirs(args.out, exist_ok=True)

    # Save normalized dataset
    output_path = os.path.join(args.out, "data.parquet")
    df.to_parquet(output_path)

    print("Rows after normalization:", len(df))


if __name__ == "__main__":
    main()