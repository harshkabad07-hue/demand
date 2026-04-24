# Retail Product Demand Prediction System

A beginner-friendly final-year college project that combines machine learning with a Flask web application to predict retail product demand.

## Project Structure

```text
retail/
|-- app.py
|-- model.py
|-- model.pkl
|-- requirements.txt
|-- README.md
|-- data/
|   `-- retail_demand_data.csv
|-- static/
|   |-- script.js
|   `-- style.css
`-- templates/
    |-- dashboard.html
    |-- index.html
    `-- prediction.html
```

## Features

- Synthetic retail dataset with 900 rows
- Data preprocessing for missing values and categorical encoding
- Random Forest Regressor as the main prediction model
- Linear Regression for comparison
- Evaluation using MAE and RMSE
- Saved trained model in `model.pkl`
- Flask API endpoint at `/predict`
- Responsive UI with animations, charts, and demand classification

## How to Run

1. Create a virtual environment:

   ```powershell
   python -m venv .venv
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

4. Start the Flask app:

   ```powershell
   python app.py
   ```

5. Open the local server shown in the terminal, usually:

   [http://127.0.0.1:5000](http://127.0.0.1:5000)

## Input Fields for Prediction

- Category
- Price
- Past Sales
- Season
- Discount

## Demand Classification

- Less than 50: Low
- 50 to 100: Medium
- Greater than 100: High
