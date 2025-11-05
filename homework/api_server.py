from pathlib import Path
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load model at startup
MODEL_PATH = Path(__file__).resolve().parent / "house_predictor.pkl"
_model_bundle = None


def get_model_bundle():
    global _model_bundle
    if _model_bundle is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Train the model first.")
        _model_bundle = joblib.load(MODEL_PATH)
    return _model_bundle


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    bundle = get_model_bundle()
    pipeline = bundle["pipeline"]
    feature_names = bundle["features"]

    payload = request.get_json(silent=True) or {}

    # Accept either a single object or a list of objects
    inputs = payload if isinstance(payload, list) else [payload]

    # Build DataFrame with expected columns; missing values become NaN
    df = pd.DataFrame(inputs)
    # Coerce types
    for col in feature_names:
        if col not in df.columns:
            df[col] = None
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[feature_names]

    # Predict; scikit-learn will error if NaNs exist
    if df.isna().any().any():
        return jsonify({
            "error": "Missing or invalid fields. Required features: " + ", ".join(feature_names)
        }), 400

    preds = pipeline.predict(df)
    # Return list if multiple, single value if one
    if len(preds) == 1:
        return jsonify({"predicted_price": float(preds[0])})
    return jsonify({"predicted_price": [float(p) for p in preds]})


if __name__ == "__main__":
    # Default Flask dev server
    app.run(host="0.0.0.0", port=5000, debug=True)
