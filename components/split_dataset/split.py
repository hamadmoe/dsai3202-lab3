import argparse
import os
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--train_out", type=str, required=True)
    parser.add_argument("--val_out", type=str, required=True)
    parser.add_argument("--test_out", type=str, required=True)
    parser.add_argument("--deployment_out", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    # ---------------- LOAD ----------------
    # Azure passes a folder → read the parquet inside
    input_path = os.path.join(args.data, "data.parquet")
    df = pd.read_parquet(input_path)

    print("Columns in dataset:", df.columns.tolist())

    # ---------------- HANDLE REVIEW YEAR ----------------
    if "review_year" in df.columns:
        year_col = "review_year"
    elif "review_year_x" in df.columns:
        year_col = "review_year_x"
    elif "review_year_y" in df.columns:
        year_col = "review_year_y"
    else:
        raise ValueError("No review_year column found in dataset.")

    print("Using year column:", year_col)

    # ---------------- SORT BY TIME ----------------
    df = df.sort_values(by=year_col)

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