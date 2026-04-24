function setDiscountLabel() {
    const discountInput = document.getElementById("discount-input");
    const discountValue = document.getElementById("discount-value");

    if (!discountInput || !discountValue) {
        return;
    }

    const syncLabel = () => {
        discountValue.textContent = `${discountInput.value}%`;
    };

    discountInput.addEventListener("input", syncLabel);
    syncLabel();
}

function renderPredictionResult(result) {
    const resultCard = document.getElementById("result-card");
    const badgeClass = result.demand_level.toLowerCase();

    resultCard.innerHTML = `
        <p class="muted">Predicted product demand</p>
        <div class="prediction-number">${result.predicted_demand}</div>
        <span class="badge ${badgeClass}">${result.demand_level} Demand</span>
        <p class="muted">This prediction combines category, price, historical sales, season, and discount impact.</p>
    `;
}

function showPredictionError(message) {
    const errorBox = document.getElementById("error-box");
    if (!errorBox) {
        return;
    }
    errorBox.textContent = message;
    errorBox.classList.remove("hidden");
}

function hidePredictionError() {
    const errorBox = document.getElementById("error-box");
    if (!errorBox) {
        return;
    }
    errorBox.classList.add("hidden");
}

function setupPredictionForm() {
    const form = document.getElementById("prediction-form");
    if (!form) {
        return;
    }

    const button = form.querySelector("button");
    const buttonText = form.querySelector(".btn-text");
    const loader = form.querySelector(".loader");

    form.addEventListener("submit", async (event) => {
        event.preventDefault();
        hidePredictionError();

        const formData = new FormData(form);
        const payload = Object.fromEntries(formData.entries());

        button.disabled = true;
        buttonText.textContent = "Predicting...";
        loader.classList.remove("hidden");

        try {
            const response = await fetch("/predict", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify(payload),
            });

            const result = await response.json();
            if (!response.ok) {
                throw new Error(result.error || "Unable to predict demand.");
            }

            renderPredictionResult(result);
        } catch (error) {
            showPredictionError(error.message);
        } finally {
            button.disabled = false;
            buttonText.textContent = "Predict Demand";
            loader.classList.add("hidden");
        }
    });
}

function renderMetrics(metrics) {
    const metricsGrid = document.getElementById("metrics-grid");
    if (!metricsGrid) {
        return;
    }

    const cards = [
        { label: "RF MAE", value: metrics.random_forest.mae.toFixed(2) },
        { label: "RF RMSE", value: metrics.random_forest.rmse.toFixed(2) },
        { label: "LR MAE", value: metrics.linear_regression.mae.toFixed(2) },
        { label: "LR RMSE", value: metrics.linear_regression.rmse.toFixed(2) },
    ];

    metricsGrid.innerHTML = cards
        .map(
            (card) => `
                <div class="metric-card">
                    <span class="muted">${card.label}</span>
                    <strong>${card.value}</strong>
                </div>
            `
        )
        .join("");
}

function setupDashboard() {
    const salesCanvas = document.getElementById("salesDemandChart");
    const seasonCanvas = document.getElementById("seasonChart");

    if (!salesCanvas || !seasonCanvas || typeof Chart === "undefined") {
        return;
    }

    fetch("/dashboard-data")
        .then((response) => response.json())
        .then((data) => {
            renderMetrics(data.metrics);

            new Chart(salesCanvas, {
                type: "bar",
                data: {
                    labels: data.sales_vs_demand.labels,
                    datasets: [
                        {
                            label: "Past Sales",
                            data: data.sales_vs_demand.past_sales,
                            backgroundColor: "rgba(15, 118, 110, 0.65)",
                            borderRadius: 8,
                        },
                        {
                            label: "Demand",
                            data: data.sales_vs_demand.demand,
                            backgroundColor: "rgba(239, 108, 61, 0.72)",
                            borderRadius: 8,
                        },
                    ],
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                },
            });

            new Chart(seasonCanvas, {
                type: "line",
                data: {
                    labels: data.seasonal_trends.labels,
                    datasets: [
                        {
                            label: "Average Demand",
                            data: data.seasonal_trends.demand,
                            borderColor: "#ef6c3d",
                            backgroundColor: "rgba(239, 108, 61, 0.16)",
                            fill: true,
                            tension: 0.35,
                        },
                    ],
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                },
            });
        })
        .catch((error) => {
            const metricsGrid = document.getElementById("metrics-grid");
            if (metricsGrid) {
                metricsGrid.innerHTML = `<div class="error-box">${error.message}</div>`;
            }
        });
}

document.addEventListener("DOMContentLoaded", () => {
    setDiscountLabel();
    setupPredictionForm();
    setupDashboard();
});
