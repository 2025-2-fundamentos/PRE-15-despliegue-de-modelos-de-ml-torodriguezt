import argparse
from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

FEATURES = [
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "waterfront",
    "condition",
]
TARGET = "price"


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Ensure correct dtypes
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    # Drop rows with missing values in used cols
    df = df.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)
    return df


def build_pipeline() -> Pipeline:
    numeric_features = FEATURES
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
        ],
        remainder="drop",
    )
    model = LinearRegression()
    pipe = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model),
    ])
    return pipe


def train(csv_path: Path, out_model_path: Path) -> None:
    print(f"Loading data from: {csv_path}")
    df = load_data(csv_path)
    X = df[FEATURES]
    y = df[TARGET]

    print(f"Training on {len(df)} rows with features: {FEATURES}")
    pipeline = build_pipeline()
    pipeline.fit(X, y)

    out_model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "features": FEATURES}, out_model_path)
    print(f"Model saved to: {out_model_path}")


def main():
    repo_root = Path(__file__).resolve().parents[1]
    default_csv = repo_root / "files" / "input" / "house_data.csv"
    default_out = Path(__file__).resolve().parent / "house_predictor.pkl"

    parser = argparse.ArgumentParser(description="Train house price predictor")
    parser.add_argument("--data", type=str, default=str(default_csv), help="Path to house_data.csv")
    parser.add_argument("--out", type=str, default=str(default_out), help="Output path for the trained model (.pkl)")
    args = parser.parse_args()

    train(Path(args.data), Path(args.out))


if __name__ == "__main__":
    main()
