import pickle
import os
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template, request

from model import DATASET_PATH, MODEL_PATH, generate_dataset, train_and_save_model


app = Flask(__name__)


def ensure_artifacts():
    if not DATASET_PATH.exists():
        generate_dataset()
    if not MODEL_PATH.exists():
        train_and_save_model()


def load_bundle():
    ensure_artifacts()
    with MODEL_PATH.open("rb") as model_file:
        return pickle.load(model_file)


def classify_demand(value: float) -> str:
    if value < 50:
        return "Low"
    if value <= 100:
        return "Medium"
    return "High"


@app.route("/")
def home():
    bundle = load_bundle()
    return render_template("index.html", categories=bundle["categories"], seasons=bundle["seasons"])


@app.route("/prediction")
def prediction_page():
    bundle = load_bundle()
    return render_template("prediction.html", categories=bundle["categories"], seasons=bundle["seasons"])


@app.route("/dashboard")
def dashboard_page():
    bundle = load_bundle()
    return render_template("dashboard.html", categories=bundle["categories"], seasons=bundle["seasons"])


@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(silent=True) or request.form
        category = payload.get("category")
        price = float(payload.get("price", 0))
        past_sales = float(payload.get("past_sales", 0))
        season = payload.get("season")
        discount = float(payload.get("discount", 0))

        if not category or not season:
            return jsonify({"error": "Category and season are required."}), 400
        if price <= 0 or past_sales < 0 or discount < 0 or discount > 50:
            return jsonify(
                {"error": "Price must be positive, past sales cannot be negative, and discount must be between 0 and 50."}
            ), 400

        bundle = load_bundle()
        features = pd.DataFrame(
            [
                {
                    "category": category,
                    "price": price,
                    "past_sales": past_sales,
                    "season": season,
                    "discount": discount,
                }
            ]
        )

        prediction = float(bundle["model"].predict(features)[0])
        rounded_prediction = round(prediction, 2)
        return jsonify(
            {
                "predicted_demand": rounded_prediction,
                "demand_level": classify_demand(rounded_prediction),
            }
        )
    except ValueError:
        return jsonify({"error": "Please enter valid numeric values."}), 400
    except Exception as exc:
        return jsonify({"error": f"Prediction failed: {exc}"}), 500


@app.route("/dashboard-data")
def dashboard_data():
    ensure_artifacts()
    dataframe = pd.read_csv(DATASET_PATH)
    dataframe["season"] = dataframe["season"].fillna("unknown")

    sales_vs_demand = dataframe[["past_sales", "demand"]].fillna(0).head(20)
    seasonal_trends = (
        dataframe.groupby("season", dropna=False)["demand"]
        .mean()
        .round(2)
        .reindex(["summer", "winter", "festival", "unknown"], fill_value=0)
    )

    bundle = load_bundle()
    return jsonify(
        {
            "sales_vs_demand": {
                "labels": [f"Item {idx + 1}" for idx in range(len(sales_vs_demand))],
                "past_sales": sales_vs_demand["past_sales"].round(2).tolist(),
                "demand": sales_vs_demand["demand"].round(2).tolist(),
            },
            "seasonal_trends": {
                "labels": seasonal_trends.index.tolist(),
                "demand": seasonal_trends.tolist(),
            },
            "metrics": bundle["metrics"],
        }
    )


if __name__ == "__main__":
    ensure_artifacts()
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "False").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
