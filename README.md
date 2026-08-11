# India Retail Demand Forecasting & Inventory Optimization Website

A resume-ready data analysis and machine learning website for Indian retail demand forecasting. The project predicts product demand by region, city, category, season, discount, stock level, and supplier lead time, then converts the forecast into an inventory decision.

## Why This Project Is Useful for Jobs

- Shows Python, Pandas, scikit-learn, Flask, and dashboard skills
- Uses India-focused retail signals such as states, cities, INR pricing, festival season, monsoon season, and store formats
- Includes business KPIs, model metrics, regional demand analysis, revenue mix, and inventory risk
- Produces decision-focused outputs: predicted demand, demand level, stockout/overstock status, and reorder quantity

## Website Pages

```text
Home        - Project overview and business use case
Prediction  - Form-based demand and inventory prediction
Dashboard   - Charts for actual vs predicted demand, regional demand, revenue, seasonality, and inventory risk
```

## Tech Stack

```text
Python
Flask
Pandas
NumPy
Scikit-learn
Random Forest Regressor
Linear Regression
Chart.js
HTML, CSS, JavaScript
```

## Dataset

The app generates a local India-region retail dataset at:

```text
data/india_retail_demand_data.csv
```

Dataset fields include:

```text
date, product_id, category, state, city, store_format, season,
price, discount, past_sales, stock_on_hand, reorder_level,
lead_time_days, demand
```

The generator models Indian retail behavior across Maharashtra, Telangana, Kerala, Karnataka, Tamil Nadu, and Delhi NCR with summer, monsoon, winter, and festival seasonality.

Optional Kaggle dataset support is also included. If you download the Kaggle Indian retail-chain files, place them here:

```text
data/india_retail_chain/
```

Expected files:

```text
train_data.csv
product_prices.csv
date_to_week_id_map.csv
```

Then run `python model.py` again.

## How to Run

1. Activate the virtual environment:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

2. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

3. Train the model and generate the dataset:

   ```powershell
   python model.py
   ```

4. Start the website:

   ```powershell
   python app.py
   ```

5. Open:

   [http://127.0.0.1:5000](http://127.0.0.1:5000)

## Resume Bullet

```text
Built an India-focused retail demand forecasting and inventory optimization website using Python, Flask, Pandas, scikit-learn, and Chart.js to predict store-level demand, identify stockout/overstock risk, and recommend reorder quantities across Indian regions, cities, categories, and seasons.
```

## Model Performance

Current generated dataset performance:

```text
Random Forest MAE: 14.06
Random Forest RMSE: 18.11
Random Forest R2: 0.897
```
