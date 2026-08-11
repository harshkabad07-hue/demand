function setupCitySelect() {
    const stateSelect = document.getElementById("state-select");
    const citySelect = document.getElementById("city-select");
    const cityDataElement = document.getElementById("city-data");

    if (!stateSelect || !citySelect || !cityDataElement) {
        return;
    }

    const cityData = JSON.parse(cityDataElement.textContent);

    const syncCities = () => {
        const cities = cityData[stateSelect.value] || [];
        citySelect.innerHTML = cities.map((city) => `<option value="${city}">${city}</option>`).join("");
    };

    stateSelect.addEventListener("change", syncCities);
    syncCities();
}

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
    const riskClass = result.inventory_status.toLowerCase().replaceAll(" ", "-");

    resultCard.innerHTML = `
        <p class="muted">Forecasted demand for the selected Indian retail context</p>
        <div class="prediction-number">${result.predicted_demand} units</div>
        <div class="result-actions">
            <span class="badge ${badgeClass}">${result.demand_level} Demand</span>
            <span class="badge ${riskClass}">${result.inventory_status}</span>
        </div>
        <div class="recommendation-box">
            <span class="muted">Recommended reorder quantity</span>
            <strong>${result.recommended_reorder_qty} units</strong>
        </div>
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
        { label: "Random Forest MAE", value: metrics.random_forest.mae.toFixed(2) },
        { label: "Random Forest RMSE", value: metrics.random_forest.rmse.toFixed(2) },
        { label: "Random Forest R2", value: metrics.random_forest.r2.toFixed(3) },
        { label: "Linear Regression RMSE", value: metrics.linear_regression.rmse.toFixed(2) },
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

function renderKpis(kpis) {
    const kpiGrid = document.getElementById("kpi-grid");
    if (!kpiGrid) {
        return;
    }

    const cards = [
        { label: "Dataset Rows", value: kpis.records.toLocaleString("en-IN") },
        { label: "Avg Demand", value: `${kpis.avg_demand} units` },
        { label: "Avg Stock", value: `${kpis.avg_stock} units` },
        { label: "Indian Regions", value: kpis.states },
    ];

    kpiGrid.innerHTML = cards
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

function makeChart(canvas, config) {
    if (!canvas || typeof Chart === "undefined") {
        return;
    }

    new Chart(canvas, {
        ...config,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: {
                        boxWidth: 12,
                        font: { family: "Manrope" },
                    },
                },
            },
            scales: config.type === "doughnut" ? undefined : {
                x: { ticks: { maxRotation: 35, minRotation: 0 } },
                y: { beginAtZero: true },
            },
            ...config.options,
        },
    });
}

function setupDashboard() {
    const salesCanvas = document.getElementById("salesDemandChart");
    const seasonCanvas = document.getElementById("seasonChart");
    const stateCanvas = document.getElementById("stateChart");
    const categoryCanvas = document.getElementById("categoryChart");
    const inventoryCanvas = document.getElementById("inventoryChart");

    if (!salesCanvas || !seasonCanvas || typeof Chart === "undefined") {
        return;
    }

    fetch("/dashboard-data")
        .then((response) => response.json())
        .then((data) => {
            renderKpis(data.kpis);
            renderMetrics(data.metrics);

            makeChart(salesCanvas, {
                type: "bar",
                data: {
                    labels: data.sales_vs_demand.labels,
                    datasets: [
                        {
                            label: "Actual Demand",
                            data: data.sales_vs_demand.demand,
                            backgroundColor: "rgba(30, 64, 175, 0.72)",
                            borderRadius: 6,
                        },
                        {
                            label: "Predicted Demand",
                            data: data.sales_vs_demand.predicted_demand,
                            backgroundColor: "rgba(217, 119, 6, 0.72)",
                            borderRadius: 6,
                        },
                    ],
                },
            });

            makeChart(stateCanvas, {
                type: "bar",
                data: {
                    labels: data.state_demand.labels,
                    datasets: [
                        {
                            label: "Total Demand",
                            data: data.state_demand.demand,
                            backgroundColor: "rgba(13, 148, 136, 0.75)",
                            borderRadius: 8,
                        },
                    ],
                },
            });

            makeChart(categoryCanvas, {
                type: "bar",
                data: {
                    labels: data.category_revenue.labels,
                    datasets: [
                        {
                            label: "Revenue in INR",
                            data: data.category_revenue.revenue,
                            backgroundColor: "rgba(190, 24, 93, 0.68)",
                            borderRadius: 8,
                        },
                    ],
                },
            });

            makeChart(seasonCanvas, {
                type: "line",
                data: {
                    labels: data.seasonal_trends.labels,
                    datasets: [
                        {
                            label: "Average Demand",
                            data: data.seasonal_trends.demand,
                            borderColor: "#d97706",
                            backgroundColor: "rgba(217, 119, 6, 0.16)",
                            fill: true,
                            tension: 0.35,
                        },
                    ],
                },
            });

            makeChart(inventoryCanvas, {
                type: "doughnut",
                data: {
                    labels: data.inventory_counts.labels,
                    datasets: [
                        {
                            data: data.inventory_counts.counts,
                            backgroundColor: ["#dc2626", "#0d9488", "#d97706"],
                            borderWidth: 0,
                        },
                    ],
                },
                options: {
                    cutout: "58%",
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
    setupCitySelect();
    setDiscountLabel();
    setupPredictionForm();
    setupDashboard();
});
