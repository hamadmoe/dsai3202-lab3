import argparse
import os
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--train_out", type=str, required=True)
    parser.add_argument("--val_out", type=str, required=True)
    parser.add_argument("--test_out", type=str, required=True)
    parser.add_argument("--deployment_out", type=str, required=True)  # 🔥 NEW
    return parser.parse_args()


def main():
    args = parse_args()

    # ---------------- LOAD ----------------
    df = pd.read_parquet(args.data)

    if "review_year" not in df.columns:
        raise ValueError("Column 'review_year' not found.")

    # ---------------- SORT BY TIME ----------------
    df = df.sort_values(by="review_year")

    total_len = len(df)

    # ---------------- DEPLOYMENT (last 10%) ----------------
    deployment_size = int(0.10 * total_len)
    deployment_df = df.iloc[-deployment_size:]

    # Remaining data
    remaining_df = df.iloc[:-deployment_size]
    remaining_len = len(remaining_df)

    # ---------------- TRAIN / VAL / TEST ----------------
    train_size = int(0.60 * remaining_len)
    val_size = int(0.15 * remaining_len)
    test_size = remaining_len - train_size - val_size

    train_df = remaining_df.iloc[:train_size]
    val_df = remaining_df.iloc[train_size:train_size + val_size]
    test_df = remaining_df.iloc[train_size + val_size:]

    # ---------------- SAVE ----------------
    os.makedirs(args.train_out, exist_ok=True)
    os.makedirs(args.val_out, exist_ok=True)
    os.makedirs(args.test_out, exist_ok=True)
    os.makedirs(args.deployment_out, exist_ok=True)

    train_df.to_parquet(os.path.join(args.train_out, "data.parquet"))
    val_df.to_parquet(os.path.join(args.val_out, "data.parquet"))
    test_df.to_parquet(os.path.join(args.test_out, "data.parquet"))
    deployment_df.to_parquet(os.path.join(args.deployment_out, "data.parquet"))

    # ---------------- PRINT ----------------
    print("Train rows:", len(train_df))
    print("Validation rows:", len(val_df))
    print("Test rows:", len(test_df))
    print("Deployment rows:", len(deployment_df))
    print("Total rows:", total_len)


if __name__ == "__main__":
    main()