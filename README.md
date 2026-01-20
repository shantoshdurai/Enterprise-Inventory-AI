# 🚀 Enterprise Inventory AI: Demand Forecasting Suite

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Architecture-o9_Solutions_Grade-orange.svg)](https://o9solutions.com/)
[![License](https://img.shields.io/badge/Presentation-Hackathon_Ready-green.svg)](#)

A production-grade **Machine Learning solution for Inventory Forecasting**, engineered to enterprise standards (o9 Solutions / Blue Yonder style). This suite features a multi-model ensemble engine, deep sequential learning, and a real-time **Command Center** for supply chain planners.

---

## 🌟 Key Features

### 🧠 Ensemble Intelligence
Combines high-performance regressors including **XGBoost, LightGBM, CatBoost, and Random Forest** to deliver stabilized, high-accuracy demand predictions.

### 📉 Deep Sequential Analysis
Leverages **LSTM (Long Short-Term Memory)** neural networks to capture complex, non-linear temporal dependencies and long-term trends in sales data.

### 📅 Structural Seasonality
Integrated **Meta Prophet** and **Auto-ARIMA** foundations to model recurring seasonal patterns, holiday effects, and structural changes.

### 🕹️ Interactive "What-If" Simulator
Real-time planning sandbox. Planners can adjust **Promotion Intensity** and **Price Points** via sliders to see instant demand re-forecasting on the dashboard.

### 🔍 Explainable AI (XAI)
Transparency via **SHAP (SHapley Additive exPlanations)** values. The system explains *why* a forecast was generated, identifying key drivers like weather, discounts, or historical lags.

---

## 🏗 System Architecture

1.  **Unified Data Pipeline**: 30+ engineered features including recursive lags, rolling statistics, cyclical time encoding, and competitor price indexes.
2.  **Hybrid Model Engine**: A modular training orchestrator that manages the lifecycle of ML, Deep Learning, and Statistical models.
3.  **Planner's Command Center**: A sleek, dark-themed **Plotly Dash** interface with glassmorphism UI elements for real-time monitoring and scenario testing.

---

## 🚀 Getting Started

### 1. Environment Setup
```bash
# Install enterprise-grade dependencies
pip install -r requirements.txt
```

### 2. Execute Training Cycle
This orchestrator processes raw data, engineers features, and trains the entire model suite.
```bash
python engine.py
```

### 3. Launch Command Center
Launch the interactive dashboard to visualize forecasts and run simulations.
```bash
python dashboard/app.py
```
> **Access URL**: `http://127.0.0.1:8050`

---

## 📊 Hackathon Demo Script (Win the Judges)

1.  **Executive Overview**: Point to the 4 KPI cards: *Current Inventory, 30d Forecast, Stockout Risk,* and *AI-Suggested Action*.
2.  **The Digital Twin**: Show how the **"AI Forecast"** (Blue-Dashed Line) tracks future demand vs the grey historical actuals.
3.  **Live Simulation (The "Killer" Feature)**: 
    *   *Promo Scenario*: Slide the **Promo slider** to 30%. Watch the blue line surge visually.
    *   *Price Elasticity*: Increase the **Price slider**. Observe the demand cooling down in real-time.
4.  **Explainability Display**: Click the SHAP breakdown. Explain: *"The AI isn't a black box; it shows that 'Promo Intensity' is currently the #1 driver for this SKU."*
5.  **Benchmark Leaderboard**: Show the Performance Chart. Prove how the **Ensemble** approach beats individual models.

---

## 🛠 Tech Stack

*   **Modeling**: Scikit-Learn, XGBoost, LightGBM, CatBoost, TensorFlow (Keras), Meta Prophet, pmdarima.
*   **Data Processing**: Pandas, NumPy, Scipy, Joblib.
*   **Interface**: Plotly Dash, Dash Bootstrap Components.
*   **Explainability**: SHAP (Shapley Values).

---

Developed for **Enterprise-Inventory-AI** 🚀

---

## 📅 Project Status

**Last Updated:** January 20, 2026  
**Status:** Active Development  
**Version:** 1.2.0

### Recent Updates
- Enhanced ensemble model accuracy
- Improved dashboard responsiveness
- Optimized data pipeline performance
- Added comprehensive documentation
