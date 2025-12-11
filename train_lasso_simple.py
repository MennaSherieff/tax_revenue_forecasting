import pathlib
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------- CONFIG ---------- #
DATA_PATH = "./data/merged_data.csv"  # <-- change if your file has a different name
MODELS_DIR = pathlib.Path("models")
MODELS_DIR.mkdir(exist_ok=True)


def main():
    # 1. Load data
    df = pd.read_csv(DATA_PATH)

    # Keep only the needed columns
    cols = ["gdp", "cpi", "exports", "imports", "tax_revenue"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in data: {missing}")

    X = df[["gdp", "cpi", "exports", "imports"]]
    y = df["tax_revenue"]

    # 2. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Build a simple pipeline: scale + Lasso
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("lasso", Lasso(alpha=0.1, random_state=42)),
        ]
    )

    # 4. Train
    pipeline.fit(X_train, y_train)

    # 5. Evaluate (just to see it works)
    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Test MSE: {mse:.4f}")
    print(f"Test R²:  {r2:.4f}")

    # 6. Save the trained pipeline
    model_path = MODELS_DIR / "lasso.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)

    print(f"Saved model to {model_path.resolve()}")


if __name__ == "__main__":
    main()
