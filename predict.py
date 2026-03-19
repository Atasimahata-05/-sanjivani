import argparse
import joblib


def main(args):
    model = joblib.load(args.model_path)
    if args.text:
        X = [args.text]
    else:
        print("Enter medicine info text (e.g., name and uses), then press Enter:")
        X = [input().strip()]

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        top_i = int(probs.argmax())
        top_prob = float(probs[top_i])
        top_class = model.classes_[top_i]

        if top_prob < args.confidence_threshold:
            print("Prediction uncertain: low confidence")
            print("Suggested output: OTHER or UNCERTAIN")
            print("Top predictions:")
            for c, p in sorted(zip(model.classes_, probs), key=lambda x: x[1], reverse=True)[:3]:
                print(f" - {c}: {p:.4f}")
            return

        print("Predicted:", top_class)
        print("Confidence:", top_prob)
        print("All class probabilities:")
        for c, p in zip(model.classes_, probs):
            print(f" - {c}: {p:.4f}")
    else:
        pred = model.predict(X)
        print("Predicted:", pred[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict therapeutic class from text with trained model")
    parser.add_argument("--model-path", type=str, default="medicine_therapeutic_model.joblib", help="Trained model path")
    parser.add_argument("--text", type=str, default=None, help="Text input to classify")
    parser.add_argument("--confidence-threshold", type=float, default=0.5, help="Minimum probability to trust a class prediction")
    args = parser.parse_args()
    main(args)
