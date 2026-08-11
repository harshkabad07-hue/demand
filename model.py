import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATASET_PATH = DATA_DIR / "india_retail_demand_data.csv"
LEGACY_DATASET_PATH = DATA_DIR / "retail_demand_data.csv"
KAGGLE_DATA_DIR = DATA_DIR / "india_retail_chain"
MODEL_PATH = BASE_DIR / "model.pkl"

STATES = ["Maharashtra", "Telangana", "Kerala", "Karnataka", "Tamil Nadu", "Delhi NCR"]
CITIES_BY_STATE = {
    "Maharashtra": ["Mumbai", "Pune", "Nagpur"],
    "Telangana": ["Hyderabad", "Warangal"],
    "Kerala": ["Kochi", "Thiruvananthapuram", "Kozhikode"],
    "Karnataka": ["Bengaluru", "Mysuru"],
    "Tamil Nadu": ["Chennai", "Coimbatore"],
    "Delhi NCR": ["Delhi", "Gurugram", "Noida"],
}
CATEGORIES = ["FMCG", "Perishables", "Staples", "Personal Care", "Home Care", "Electronics"]
STORE_FORMATS = ["Supermarket", "Hypermarket", "Convenience Store", "Online"]
SEASONS = ["summer", "monsoon", "winter", "festival"]

FEATURE_COLUMNS = [
    "category",
    "state",
    "city",
    "store_format",
    "season",
    "price",
    "discount",
    "past_sales",
    "stock_on_hand",
    "reorder_level",
    "lead_time_days",
]
TARGET_COLUMN = "demand"


def detect_season(date_value) -> str:
    month = pd.to_datetime(date_value).month
    if month in [3, 4, 5, 6]:
        return "summer"
    if month in [7, 8, 9]:
        return "monsoon"
    if month in [10, 11]:
        return "festival"
    return "winter"


def generate_dataset(file_path: Path = DATASET_PATH, rows: int = 2400, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    category_effect = {
        "FMCG": 42,
        "Perishables": 48,
        "Staples": 38,
        "Personal Care": 26,
        "Home Care": 22,
        "Electronics": 14,
    }
    season_effect = {"summer": 12, "monsoon": 8, "winter": 16, "festival": 34}
    state_effect = {
        "Maharashtra": 18,
        "Telangana": 13,
        "Kerala": 10,
        "Karnataka": 16,
        "Tamil Nadu": 14,
        "Delhi NCR": 15,
    }
    format_effect = {
        "Supermarket": 16,
        "Hypermarket": 24,
        "Convenience Store": 8,
        "Online": 18,
    }
    price_ranges = {
        "FMCG": (35, 450),
        "Perishables": (20, 280),
        "Staples": (45, 650),
        "Personal Care": (60, 900),
        "Home Care": (80, 750),
        "Electronics": (650, 28000),
    }

    dates = pd.date_range("2023-01-01", "2025-12-31", freq="D")
    records = []

    for index in range(rows):
        date = rng.choice(dates)
        state = str(rng.choice(STATES))
        city = str(rng.choice(CITIES_BY_STATE[state]))
        category = str(rng.choice(CATEGORIES, p=[0.23, 0.18, 0.18, 0.15, 0.14, 0.12]))
        store_format = str(rng.choice(STORE_FORMATS, p=[0.34, 0.22, 0.26, 0.18]))
        season = detect_season(date)
        low_price, high_price = price_ranges[category]
        price = round(float(rng.uniform(low_price, high_price)), 2)
        discount = int(rng.integers(0, 41))
        lead_time_days = int(rng.integers(2, 11))
        past_sales = int(rng.integers(20, 320))
        reorder_level = int(rng.integers(45, 180))
        stock_on_hand = int(rng.integers(20, 420))

        festival_boost = 1.12 if season == "festival" and category in ["FMCG", "Staples", "Electronics"] else 1
        monsoon_penalty = 0.92 if season == "monsoon" and category == "Perishables" else 1
        discount_boost = discount * rng.uniform(0.85, 1.35)
        price_penalty = price * (0.006 if category == "Electronics" else 0.035)
        stock_pressure = max(0, reorder_level - stock_on_hand) * 0.16
        noise = rng.normal(0, 13)

        demand = (
            28
            + (past_sales * 0.46)
            + category_effect[category]
            + season_effect[season]
            + state_effect[state]
            + format_effect[store_format]
            + discount_boost
            - price_penalty
            + stock_pressure
            + noise
        )
        demand = max(8, round(float(demand * festival_boost * monsoon_penalty), 2))

        records.append(
            {
                "date": pd.Timestamp(date).strftime("%Y-%m-%d"),
                "product_id": f"IN-SKU-{10000 + index}",
                "category": category,
                "state": state,
                "city": city,
                "store_format": store_format,
                "season": season,
                "price": price,
                "discount": discount,
                "past_sales": past_sales,
                "stock_on_hand": stock_on_hand,
                "reorder_level": reorder_level,
                "lead_time_days": lead_time_days,
                "demand": demand,
            }
        )

    dataframe = pd.DataFrame(records)

    for column, frac in {"price": 0.02, "past_sales": 0.015, "season": 0.01}.items():
        missing_indices = dataframe.sample(frac=frac, random_state=seed + len(column)).index
        dataframe.loc[missing_indices, column] = np.nan

    dataframe.to_csv(file_path, index=False)
    return dataframe


def load_kaggle_indian_retail_dataset() -> pd.DataFrame | None:
    train_path = KAGGLE_DATA_DIR / "train_data.csv"
    prices_path = KAGGLE_DATA_DIR / "product_prices.csv"
    weeks_path = KAGGLE_DATA_DIR / "date_to_week_id_map.csv"

    if not train_path.exists():
        return None

    dataframe = pd.read_csv(train_path)
    dataframe["date"] = pd.to_datetime(dataframe["date"])
    dataframe["season"] = dataframe["date"].apply(detect_season)
    dataframe["category"] = dataframe["category_of_product"].astype(str)
    dataframe["state"] = dataframe["state"].astype(str)
    dataframe["city"] = dataframe["state"].map(
        {
            "Maharashtra": "Mumbai",
            "Telangana": "Hyderabad",
            "Kerala": "Kochi",
        }
    ).fillna("Unknown")
    dataframe["store_format"] = "Retail Outlet"

    if prices_path.exists() and weeks_path.exists():
        prices = pd.read_csv(prices_path)
        weeks = pd.read_csv(weeks_path)
        dataframe = dataframe.merge(weeks, on="date", how="left")
        dataframe = dataframe.merge(
            prices,
            on=["outlet", "product_identifier", "week_id"],
            how="left",
        )
        price_column = "sell_price" if "sell_price" in dataframe.columns else "price"
        dataframe["price"] = dataframe[price_column]
    else:
        dataframe["price"] = np.nan

    dataframe = dataframe.sort_values(["outlet", "product_identifier", "date"])
    dataframe["past_sales"] = (
        dataframe.groupby(["outlet", "product_identifier"])["sales"]
        .shift(1)
        .fillna(dataframe["sales"].median())
    )
    dataframe["discount"] = 0
    dataframe["stock_on_hand"] = (dataframe["past_sales"] * 1.35).round().clip(lower=10)
    dataframe["reorder_level"] = (dataframe["past_sales"] * 0.85).round().clip(lower=5)
    dataframe["lead_time_days"] = dataframe["state"].map(
        {"Maharashtra": 5, "Telangana": 6, "Kerala": 7}
    ).fillna(6)
    dataframe["demand"] = dataframe["sales"]

    clean = dataframe[["date", *FEATURE_COLUMNS, TARGET_COLUMN]].copy()
    clean["date"] = pd.to_datetime(clean["date"]).dt.strftime("%Y-%m-%d")
    return clean


def load_dataset(file_path: Path = DATASET_PATH) -> pd.DataFrame:
    kaggle_dataframe = load_kaggle_indian_retail_dataset()
    if kaggle_dataframe is not None:
        kaggle_dataframe.to_csv(file_path, index=False)
        return kaggle_dataframe

    if file_path.exists():
        return pd.read_csv(file_path)

    if LEGACY_DATASET_PATH.exists():
        LEGACY_DATASET_PATH.unlink(missing_ok=True)

    return generate_dataset(file_path=file_path)


def build_preprocessor() -> ColumnTransformer:
    numeric_features = ["price", "discount", "past_sales", "stock_on_hand", "reorder_level", "lead_time_days"]
    categorical_features = ["category", "state", "city", "store_format", "season"]

    numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
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


def classify_inventory(row: pd.Series) -> str:
    forecast = row.get("predicted_demand", row.get(TARGET_COLUMN, 0))
    stock = row.get("stock_on_hand", 0)
    reorder = row.get("reorder_level", 0)

    if stock < forecast or stock <= reorder:
        return "Stockout Risk"
    if stock > forecast * 2.2:
        return "Overstock Risk"
    return "Healthy"


def train_and_save_model() -> dict:
    dataframe = load_dataset()
    X = dataframe[FEATURE_COLUMNS]
    y = dataframe[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    random_forest_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=300,
                    random_state=42,
                    max_depth=16,
                    min_samples_split=4,
                    n_jobs=-1,
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
            "r2": float(r2_score(y_test, rf_predictions)),
        },
        "linear_regression": {
            "mae": float(mean_absolute_error(y_test, lr_predictions)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, lr_predictions))),
            "r2": float(r2_score(y_test, lr_predictions)),
        },
    }

    bundle = {
        "model": random_forest_pipeline,
        "metrics": metrics,
        "feature_columns": FEATURE_COLUMNS,
        "categories": CATEGORIES,
        "states": STATES,
        "cities_by_state": CITIES_BY_STATE,
        "store_formats": STORE_FORMATS,
        "seasons": SEASONS,
        "dataset_path": str(DATASET_PATH),
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
    print("Random Forest R2:", round(trained["metrics"]["random_forest"]["r2"], 3))
