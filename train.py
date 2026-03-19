import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib


def build_text(df):
    # Combine name/use/side effects into a single text field
    text_cols = ["name", "use0", "use1", "use2", "use3", "use4"] + [f"sideEffect{i}" for i in range(42)]
    for c in text_cols:
        if c not in df.columns:
            df[c] = ""
    df[text_cols] = df[text_cols].fillna("")
    return (df["name"].fillna("") + " ") + df["use0"].fillna("")


def main(args):
    df = pd.read_csv(args.data_path)
    df = df.dropna(subset=["Therapeutic Class"])
    df["text"] = df["name"].fillna("") + " " + df[["use0", "use1", "use2", "use3", "use4"]].fillna("").agg(" ".join, axis=1)
    df["text"] = df["text"].replace("\n", " ", regex=True)
    df["text"] = df["text"].astype(str)
    df["label"] = df["Therapeutic Class"].astype(str).fillna("Unknown")

    # Group rare classes into ONE "OTHER" category for better reliability.
    label_counts = df["label"].value_counts()
    keep_labels = set(label_counts[label_counts >= args.min_class_support].index)
    if args.max_classes is not None:
        top_labels = list(label_counts.head(args.max_classes).index)
        keep_labels = set(label_counts[label_counts >= args.min_class_support].index).union(top_labels)

    if len(keep_labels) == 0:
        keep_labels = set(label_counts.head(args.max_classes if args.max_classes is not None else 10).index)

    df["label_reduced"] = df["label"].where(df["label"].isin(keep_labels), "OTHER")

    print("Original classes:", len(label_counts))
    print("Kept labels:", len(keep_labels), "+ OTHER")
    print(df["label_reduced"].value_counts().to_string())

    # Avoid stratify fail when labels have only one sample.
    min_label_count = df["label_reduced"].value_counts().min()
    if min_label_count < 2:
        df = df[df["label_reduced"].isin(df["label_reduced"].value_counts()[lambda x: x >= 2].index)]

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label_reduced"], test_size=0.2, random_state=42, stratify=df["label_reduced"]
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2), stop_words="english")),
        ("clf", LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", random_state=42)),
    ])

    print("Training model...")
    pipeline.fit(X_train, y_train)

    print("Evaluating on test set...")
    y_pred = pipeline.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    print(f"Saving model to {args.output_model}")
    joblib.dump(pipeline, args.output_model)
    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a simple therapeutic class model from medicine_dataset.csv")
    parser.add_argument("--data-path", type=str, default="medicine_dataset.csv", help="Path to medicine dataset CSV")
    parser.add_argument("--output-model", type=str, default="medicine_therapeutic_model.joblib", help="Pickle file to save trained model")
    parser.add_argument("--min-class-support", type=int, default=200, help="Minimum number of examples per class to keep without collapsing into OTHER")
    parser.add_argument("--max-classes", type=int, default=12, help="Maximum top classes to keep, others grouped as OTHER")
    args = parser.parse_args()
    main(args)
