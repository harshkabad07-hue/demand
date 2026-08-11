import os
import pickle

import pandas as pd
from flask import Flask, jsonify, render_template, request

from model import (
    DATASET_PATH,
    FEATURE_COLUMNS,
    MODEL_PATH,
    classify_inventory,
    generate_dataset,
    train_and_save_model,
)


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
    if value < 80:
        return "Low"
    if value <= 180:
        return "Medium"
    return "High"


@app.route("/")
def home():
    bundle = load_bundle()
    return render_template(
        "index.html",
        categories=bundle["categories"],
        states=bundle["states"],
        seasons=bundle["seasons"],
    )


@app.route("/prediction")
def prediction_page():
    bundle = load_bundle()
    return render_template(
        "prediction.html",
        categories=bundle["categories"],
        states=bundle["states"],
        cities_by_state=bundle["cities_by_state"],
        store_formats=bundle["store_formats"],
        seasons=bundle["seasons"],
    )


@app.route("/dashboard")
def dashboard_page():
    bundle = load_bundle()
    return render_template("dashboard.html", categories=bundle["categories"], seasons=bundle["seasons"])


@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(silent=True) or request.form
        row = {
            "category": payload.get("category"),
            "state": payload.get("state"),
            "city": payload.get("city"),
            "store_format": payload.get("store_format"),
            "season": payload.get("season"),
            "price": float(payload.get("price", 0)),
            "discount": float(payload.get("discount", 0)),
            "past_sales": float(payload.get("past_sales", 0)),
            "stock_on_hand": float(payload.get("stock_on_hand", 0)),
            "reorder_level": float(payload.get("reorder_level", 0)),
            "lead_time_days": float(payload.get("lead_time_days", 0)),
        }

        missing_fields = [field for field in FEATURE_COLUMNS if row.get(field) in [None, ""]]
        if missing_fields:
            return jsonify({"error": f"Missing required fields: {', '.join(missing_fields)}."}), 400
        if row["price"] <= 0 or row["past_sales"] < 0:
            return jsonify({"error": "Price must be positive and past sales cannot be negative."}), 400
        if row["discount"] < 0 or row["discount"] > 70:
            return jsonify({"error": "Discount must be between 0 and 70 percent."}), 400
        if row["stock_on_hand"] < 0 or row["reorder_level"] < 0 or row["lead_time_days"] <= 0:
            return jsonify({"error": "Inventory values must be valid positive numbers."}), 400

        bundle = load_bundle()
        features = pd.DataFrame([row], columns=FEATURE_COLUMNS)

        prediction = float(bundle["model"].predict(features)[0])
        rounded_prediction = round(prediction, 2)
        inventory_row = pd.Series({**row, "predicted_demand": rounded_prediction})

        return jsonify(
            {
                "predicted_demand": rounded_prediction,
                "demand_level": classify_demand(rounded_prediction),
                "inventory_status": classify_inventory(inventory_row),
                "recommended_reorder_qty": max(0, round(rounded_prediction - row["stock_on_hand"] + row["reorder_level"], 0)),
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
    bundle = load_bundle()

    sample = dataframe.sample(min(18, len(dataframe)), random_state=8).reset_index(drop=True)
    sample_predictions = bundle["model"].predict(sample[FEATURE_COLUMNS])
    sample["predicted_demand"] = sample_predictions
    sample["inventory_status"] = sample.apply(classify_inventory, axis=1)

    seasonal_trends = dataframe.groupby("season", dropna=False)["demand"].mean().round(2).reindex(bundle["seasons"], fill_value=0)
    state_demand = dataframe.groupby("state")["demand"].sum().sort_values(ascending=False).round(2)
    category_revenue = (
        dataframe.assign(revenue=dataframe["price"].fillna(dataframe["price"].median()) * dataframe["demand"])
        .groupby("category")["revenue"]
        .sum()
        .sort_values(ascending=False)
        .round(2)
    )
    inventory_counts = sample["inventory_status"].value_counts().reindex(["Stockout Risk", "Healthy", "Overstock Risk"], fill_value=0)

    return jsonify(
        {
            "sales_vs_demand": {
                "labels": [f"{row.category} - {row.city}" for row in sample.itertuples()],
                "past_sales": sample["past_sales"].round(2).tolist(),
                "demand": sample["demand"].round(2).tolist(),
                "predicted_demand": sample["predicted_demand"].round(2).tolist(),
            },
            "seasonal_trends": {
                "labels": seasonal_trends.index.tolist(),
                "demand": seasonal_trends.tolist(),
            },
            "state_demand": {
                "labels": state_demand.index.tolist(),
                "demand": state_demand.tolist(),
            },
            "category_revenue": {
                "labels": category_revenue.index.tolist(),
                "revenue": category_revenue.tolist(),
            },
            "inventory_counts": {
                "labels": inventory_counts.index.tolist(),
                "counts": inventory_counts.tolist(),
            },
            "metrics": bundle["metrics"],
            "kpis": {
                "records": int(len(dataframe)),
                "avg_demand": round(float(dataframe["demand"].mean()), 2),
                "avg_stock": round(float(dataframe["stock_on_hand"].mean()), 2),
                "states": int(dataframe["state"].nunique()),
            },
        }
    )


if __name__ == "__main__":
    ensure_artifacts()
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "False").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
