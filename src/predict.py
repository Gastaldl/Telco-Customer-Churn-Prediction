import sys
from pathlib import Path

import joblib
import pandas as pd
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess import BINARY_INT, CATEGORICAL, NUMERIC, load_clean


DATA_PATH = PROJECT_ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
PREPROCESSOR_PATH = PROJECT_ROOT / "models" / "preprocessor.joblib"
MODEL_PATH = PROJECT_ROOT / "models" / "churn_nn.keras"
OUTPUT_PATH = PROJECT_ROOT / "data" / "churn_predictions.csv"


def main() -> None:
    df = load_clean(DATA_PATH)

    features = NUMERIC + CATEGORICAL + BINARY_INT
    x = df[features]

    preprocessor = joblib.load(PREPROCESSOR_PATH)
    model = tf.keras.models.load_model(MODEL_PATH)

    x_mat = preprocessor.transform(x)
    if hasattr(x_mat, "toarray"):
        x_mat = x_mat.toarray()

    churn_prob = model.predict(x_mat).ravel()

    risk_band = pd.cut(
        churn_prob,
        bins=[-0.01, 0.50, 0.68, 1.01],
        labels=["Low", "Medium", "High"],
    )

    recommended_action = risk_band.map(
        {
            "Low": "No immediate action",
            "Medium": "Light action / monitoring",
            "High": "Direct retention action",
        }
    )

    out = df[["customerID"]].copy()
    out["churn_real"] = df["Churn"]
    out["churn_prob"] = churn_prob.round(4)
    out["risk_band"] = risk_band
    out["recommended_action"] = recommended_action
    out["risk_band_order"] = out["risk_band"].map({"Low": 1, "Medium": 2, "High": 3})
    out["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, float("inf")],
        labels=["0-12", "13-24", "25-48", "49+"],
    )
    out["tenure_group_order"] = out["tenure_group"].map(
        {"0-12": 1, "13-24": 2, "25-48": 3, "49+": 4}
    )
    out["monthly_charge_band"] = pd.cut(
        df["MonthlyCharges"],
        bins=[-0.01, 35, 70, float("inf")],
        labels=["Low", "Medium", "High"],
    )
    out["monthly_charge_band_order"] = out["monthly_charge_band"].map(
        {"Low": 1, "Medium": 2, "High": 3}
    )

    for col in ["Contract", "tenure", "MonthlyCharges", "InternetService", "PaymentMethod"]:
        out[col] = df[col].values

    out.to_csv(OUTPUT_PATH, index=False)
    print(f"OK - {len(out)} rows saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
