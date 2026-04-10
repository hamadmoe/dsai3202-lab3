import argparse
import os
import time
import azureml.mlflow
import mlflow
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier


# --------------------------------------------------
# Arguments
# --------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()


# --------------------------------------------------
# Load data
# --------------------------------------------------
def load_data(path):
    input_path = os.path.join(path, "data.parquet")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Path does not exist: {input_path}")

    # 🔥 LOAD ONLY SMALL SAMPLE (CRITICAL)
    df = pd.read_parquet(input_path)

    MAX_ROWS = 15000
    if len(df) > MAX_ROWS:
        df = df.sample(n=MAX_ROWS, random_state=42)
        print(f"Loaded and sampled {MAX_ROWS} rows")

    print("Loaded shape:", df.shape)

    return df


# --------------------------------------------------
# Labels
# --------------------------------------------------
def create_labels(df):
    if "overall" not in df.columns:
        raise RuntimeError("Column 'overall' is missing.")

    df["label"] = (df["overall"] >= 4).astype(int)
    return df


# --------------------------------------------------
# Features
# --------------------------------------------------
def build_features(df):

    # Drop non-feature columns
    drop_cols = ["asin", "reviewerID", "overall", "label"]
    feature_cols = [col for col in df.columns if col not in drop_cols]

    # Keep only numeric columns
    X = df[feature_cols].select_dtypes(include=["number"])

    if len(X.columns) == 0:
        raise RuntimeError("No numeric features found.")

    # 🔥 LIMIT FEATURES
    MAX_FEATURES = 50
    X = X.iloc[:, :MAX_FEATURES]

    # 🔥 LIMIT ROWS (CRITICAL)
    MAX_ROWS = 20000
    if len(X) > MAX_ROWS:
        X = X.sample(n=MAX_ROWS, random_state=42)
        print(f"Sampled down to {MAX_ROWS} rows")

    print("Feature matrix shape:", X.shape)

    return X


# --------------------------------------------------
# Evaluation
# --------------------------------------------------
def evaluate(model, X, y, split):
    preds = model.predict(X)
    acc = accuracy_score(y, preds)

    mlflow.log_metric(f"{split}_accuracy", acc)
    print(f"{split} accuracy:", acc)


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    args = parse_args()
    start_time = time.time()

    print("Loading data...")
    train_df = load_data(args.train_data)
    val_df = load_data(args.val_data)
    test_df = load_data(args.test_data)

    print("Creating labels...")
    train_df = create_labels(train_df)
    val_df = create_labels(val_df)
    test_df = create_labels(test_df)

    print("Building features...")
    train_df = train_df.sample(n=min(len(train_df), 20000), random_state=42)

    X_train = build_features(train_df)
    y_train = train_df["label"]

    X_val = build_features(val_df)
    y_val = val_df["label"]

    X_test = build_features(test_df)
    y_test = test_df["label"]

    print("Training model...")
    model = SGDClassifier(loss="log_loss", max_iter=1000)
    model.fit(X_train, y_train)

    print("Evaluating...")
    evaluate(model, X_train, y_train, "train")
    evaluate(model, X_val, y_val, "val")
    evaluate(model, X_test, y_test, "test")

    print("Saving model...")
    os.makedirs(args.output, exist_ok=True)
    model_path = os.path.join(args.output, "model.pkl")
    joblib.dump(model, model_path)

    mlflow.log_artifact(model_path)

    runtime = time.time() - start_time
    mlflow.log_metric("training_runtime_seconds", runtime)

    print("Done.")


if __name__ == "__main__":
    main()