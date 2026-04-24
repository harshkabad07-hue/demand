import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATASET_PATH = DATA_DIR / "retail_demand_data.csv"
MODEL_PATH = BASE_DIR / "model.pkl"

CATEGORIES = [
    "Groceries",
    "Electronics",
    "Clothing",
    "Home Decor",
    "Beauty",
    "Toys",
]
SEASONS = ["summer", "winter", "festival"]
FEATURE_COLUMNS = ["category", "price", "past_sales", "season", "discount"]
TARGET_COLUMN = "demand"


def generate_dataset(file_path: Path = DATASET_PATH, rows: int = 900, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    category_effect = {
        "Groceries": 24,
        "Electronics": 10,
        "Clothing": 18,
        "Home Decor": 12,
        "Beauty": 14,
        "Toys": 16,
    }
    season_effect = {"summer": 12, "winter": 18, "festival": 30}

    records = []
    for index in range(rows):
        category = rng.choice(CATEGORIES)
        season = rng.choice(SEASONS, p=[0.35, 0.25, 0.40])
        price = round(float(rng.uniform(8, 500)), 2)
        past_sales = int(rng.integers(15, 160))
        discount = int(rng.integers(0, 51))
        promo_boost = discount * rng.uniform(0.7, 1.3)
        price_penalty = price * rng.uniform(0.03, 0.08)
        noise = rng.normal(0, 8)

        demand = (
            18
            + (past_sales * 0.55)
            + category_effect[category]
            + season_effect[season]
            + promo_boost
            - price_penalty
            + noise
        )
        demand = max(10, round(demand, 2))

        records.append(
            {
                "product_id": f"P{1000 + index}",
                "category": category,
                "price": price,
                "past_sales": past_sales,
                "season": season,
                "discount": discount,
                "demand": demand,
            }
        )

    dataframe = pd.DataFrame(records)

    # Introduce a few missing values so preprocessing has something realistic to handle.
    for column, frac in {"price": 0.03, "past_sales": 0.02, "season": 0.02}.items():
        missing_indices = dataframe.sample(frac=frac, random_state=seed + len(column)).index
        dataframe.loc[missing_indices, column] = np.nan

    dataframe.to_csv(file_path, index=False)
    return dataframe


def load_dataset(file_path: Path = DATASET_PATH) -> pd.DataFrame:
    if not file_path.exists():
        return generate_dataset(file_path=file_path)
    return pd.read_csv(file_path)


def build_preprocessor() -> ColumnTransformer:
    numeric_features = ["price", "past_sales", "discount"]
    categorical_features = ["category", "season"]

    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def train_and_save_model() -> dict:
    dataframe = load_dataset()
    X = dataframe[FEATURE_COLUMNS]
    y = dataframe[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = build_preprocessor()

    random_forest_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=250,
                    random_state=42,
                    max_depth=14,
                    min_samples_split=4,
                ),
            ),
        ]
    )
    linear_regression_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", LinearRegression()),
        ]
    )

    random_forest_pipeline.fit(X_train, y_train)
    linear_regression_pipeline.fit(X_train, y_train)

    rf_predictions = random_forest_pipeline.predict(X_test)
    lr_predictions = linear_regression_pipeline.predict(X_test)

    metrics = {
        "random_forest": {
            "mae": float(mean_absolute_error(y_test, rf_predictions)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, rf_predictions))),
        },
        "linear_regression": {
            "mae": float(mean_absolute_error(y_test, lr_predictions)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, lr_predictions))),
        },
    }

    bundle = {
        "model": random_forest_pipeline,
        "metrics": metrics,
        "feature_columns": FEATURE_COLUMNS,
        "categories": CATEGORIES,
        "seasons": SEASONS,
    }

    with MODEL_PATH.open("wb") as model_file:
        pickle.dump(bundle, model_file)

    return bundle


if __name__ == "__main__":
    trained = train_and_save_model()
    print("Dataset saved to:", DATASET_PATH)
    print("Model saved to:", MODEL_PATH)
    print("Random Forest MAE:", round(trained["metrics"]["random_forest"]["mae"], 2))
    print("Random Forest RMSE:", round(trained["metrics"]["random_forest"]["rmse"], 2))
    print("Linear Regression MAE:", round(trained["metrics"]["linear_regression"]["mae"], 2))
    print("Linear Regression RMSE:", round(trained["metrics"]["linear_regression"]["rmse"], 2))
