from pathlib import Path
from flask import Flask, render_template_string, request
import joblib
import pandas as pd

app = Flask(__name__)

MODEL_PATH = Path(__file__).resolve().parent / "house_predictor.pkl"
_model_bundle = None


def get_model_bundle():
    global _model_bundle
    if _model_bundle is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Train the model first.")
        _model_bundle = joblib.load(MODEL_PATH)
    return _model_bundle


TEMPLATE = """
<!doctype html>
<html lang="es">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>House Price Predictor</title>
    <style>
      body { font-family: system-ui, sans-serif; margin: 2rem; }
      form { display: grid; grid-template-columns: 200px 1fr; gap: 0.5rem 1rem; max-width: 700px; }
      label { font-weight: 600; }
      input { padding: 0.4rem; }
      .row { display: contents; }
      .submit { grid-column: 1 / -1; margin-top: 1rem; }
      .result { margin-top: 1rem; font-size: 1.2rem; color: #0a7; }
      .error { margin-top: 1rem; color: #a00; }
    </style>
  </head>
  <body>
    <h1>House Price Predictor</h1>
    <form method="post">
      {% for field in fields %}
      <div class="row">
        <label for="{{ field }}">{{ field }}</label>
        <input id="{{ field }}" name="{{ field }}" type="number" step="any" required value="{{ values.get(field, '') }}" />
      </div>
      {% endfor %}
      <div class="submit">
        <button type="submit">Predecir precio</button>
      </div>
    </form>

    {% if error %}
      <div class="error">{{ error }}</div>
    {% endif %}

    {% if prediction is not none %}
      <div class="result">Precio estimado: <strong>{{ prediction | round(2) }}</strong></div>
    {% endif %}

    <hr />
    <p>API curl (ejemplo):</p>
    <pre>
    curl -X POST http://127.0.0.1:5000/predict ^
      -H "Content-Type: application/json" ^
      -d "{\"bedrooms\":3,\"bathrooms\":2,\"sqft_living\":1800,\"sqft_lot\":5000,\"floors\":1,\"waterfront\":0,\"condition\":3}"
    </pre>
  </body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    bundle = get_model_bundle()
    pipeline = bundle["pipeline"]
    feature_names = bundle["features"]

    values = {}
    pred = None
    error = None

    if request.method == "POST":
        try:
            values = {f: request.form.get(f, type=float) for f in feature_names}
            df = pd.DataFrame([values])
            pred = float(pipeline.predict(df)[0])
        except Exception as ex:
            error = f"Error al predecir: {ex}"

    return render_template_string(TEMPLATE, fields=feature_names, values=values, prediction=pred, error=error)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
