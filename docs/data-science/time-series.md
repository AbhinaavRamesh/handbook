# Time Series Analysis

> **Sequential data forecasting and analysis** - decomposition, stationarity, ARIMA, and modern methods

---

## Overview

Time series questions appear in data science interviews for roles involving forecasting (demand, revenue, engagement), anomaly detection, and causal inference with temporal data. You need to understand both classical statistical methods and when to reach for ML-based approaches.

---

## Time Series Components

| Component | Description | Example |
|-----------|------------|--------|
| **Trend** | Long-term increase or decrease | Growing user base over years |
| **Seasonality** | Regular periodic pattern | Higher sales every December |
| **Cyclical** | Irregular long-term oscillations | Economic boom/bust cycles |
| **Residual** | Random noise after removing above | Unexplained variation |

### Decomposition

$$y_t = T_t + S_t + R_t \quad \text{(additive)}$$

$$y_t = T_t \times S_t \times R_t \quad \text{(multiplicative)}$$

| Type | When to Use |
|------|------------|
| **Additive** | Seasonal amplitude is constant over time |
| **Multiplicative** | Seasonal amplitude grows with the level |

---

## Stationarity

A time series is **stationary** if its statistical properties (mean, variance, autocorrelation) don't change over time.

### Why It Matters

Most classical forecasting methods (ARIMA) require stationarity. Non-stationary data produces spurious correlations and unreliable forecasts.

### Testing for Stationarity

| Test | Null Hypothesis | Reject Means |
|------|----------------|-------------|
| **ADF (Augmented Dickey-Fuller)** | Unit root present (non-stationary) | Series is stationary |
| **KPSS** | Series is stationary | Series is non-stationary |
| **Phillips-Perron** | Unit root present | Series is stationary |

**Best practice**: Use both ADF and KPSS together for confirmation.

### Making Data Stationary

| Technique | Removes | When to Use |
|-----------|---------|-------------|
| **Differencing** | Trend | $y'_t = y_t - y_{t-1}$ |
| **Seasonal differencing** | Seasonality | $y'_t = y_t - y_{t-m}$ |
| **Log transform** | Multiplicative seasonality | Stabilize variance |
| **Detrending** | Trend | Subtract fitted trend line |

---

## Autocorrelation

| Function | What It Shows | Used For |
|----------|-------------|----------|
| **ACF** (Autocorrelation) | Correlation with lagged values | Identify MA order ($q$) |
| **PACF** (Partial Autocorrelation) | Direct correlation (removing intermediate lags) | Identify AR order ($p$) |

### Reading ACF/PACF for ARIMA Order

| Pattern | ACF | PACF | Model |
|---------|-----|------|-------|
| AR($p$) | Decays gradually | Cuts off after lag $p$ | Autoregressive |
| MA($q$) | Cuts off after lag $q$ | Decays gradually | Moving Average |
| ARMA($p$,$q$) | Decays | Decays | Both |

---

## Classical Models

### ARIMA(p, d, q)

| Parameter | Meaning | Determined By |
|-----------|---------|---------------|
| $p$ | Autoregressive order | PACF cutoff |
| $d$ | Differencing order | Number of differences for stationarity |
| $q$ | Moving average order | ACF cutoff |

$$y'_t = c + \sum_{i=1}^p \phi_i y'_{t-i} + \sum_{j=1}^q \theta_j \epsilon_{t-j} + \epsilon_t$$

### SARIMA(p, d, q)(P, D, Q, m)

Extends ARIMA with seasonal components:
- $(P, D, Q)$: seasonal AR, differencing, and MA orders
- $m$: seasonal period (12 for monthly, 7 for daily with weekly pattern)

### Exponential Smoothing

| Method | Components | Parameters |
|--------|-----------|------------|
| **Simple (SES)** | Level | $\alpha$ |
| **Holt's** | Level + Trend | $\alpha, \beta$ |
| **Holt-Winters** | Level + Trend + Seasonality | $\alpha, \beta, \gamma$ |

$$\hat{y}_{t+1} = \alpha y_t + (1 - \alpha) \hat{y}_t \quad \text{(SES)}$$

---

## Modern / ML Approaches

| Method | Best For | Pros | Cons |
|--------|----------|------|------|
| **Prophet** | Business time series with holidays | Easy to use, handles missing data | Limited for complex patterns |
| **XGBoost / LightGBM** | Feature-rich time series | Handles non-linearity, many features | Needs careful feature engineering |
| **LSTM / GRU** | Long sequences, complex patterns | Captures long-range dependencies | Needs lots of data, slow to train |
| **Temporal Fusion Transformer** | Multi-horizon forecasting | State-of-the-art, interpretable | Complex architecture |
| **N-BEATS / N-HiTS** | Pure time series (no exogenous) | Strong baselines, fast | No external features |

### Feature Engineering for ML-Based Forecasting

| Feature Type | Examples |
|-------------|--------|
| **Lag features** | $y_{t-1}, y_{t-7}, y_{t-30}$ |
| **Rolling statistics** | 7-day mean, 30-day std |
| **Calendar features** | Day of week, month, is_holiday |
| **Fourier features** | $\sin(2\pi t / P), \cos(2\pi t / P)$ |
| **External signals** | Weather, promotions, events |

---

## Evaluation Metrics

| Metric | Formula | Properties |
|--------|---------|------------|
| **MAE** | $\frac{1}{n}\sum\|y_t - \hat{y}_t\|$ | Robust to outliers |
| **RMSE** | $\sqrt{\frac{1}{n}\sum(y_t - \hat{y}_t)^2}$ | Penalizes large errors |
| **MAPE** | $\frac{100}{n}\sum\|\frac{y_t - \hat{y}_t}{y_t}\|$ | Scale-independent, undefined at $y_t = 0$ |
| **sMAPE** | Symmetric version of MAPE | Bounded, less biased |
| **MASE** | MAE relative to naive forecast | Best for comparing across series |

### Cross-Validation for Time Series

**Never use random splits** — use time-based splits to prevent data leakage.

| Method | Description |
|--------|-------------|
| **Expanding window** | Train on [1, t], test on [t+1, t+h], then [1, t+h], etc. |
| **Sliding window** | Train on [t-w, t], test on [t+1, t+h], slide forward |
| **Walk-forward** | One-step-ahead predictions with expanding training |

---

## Interview Questions

1. **"How would you forecast daily revenue for the next 30 days?"**
   - Start with EDA: check for trend, seasonality, outliers. Try SARIMA as baseline if clear seasonal pattern. Compare with Prophet (handles holidays). For richer features (promotions, events), use gradient boosting with lag/calendar features. Evaluate with walk-forward validation using MASE.

2. **"What's the difference between ARIMA and exponential smoothing?"**
   - ARIMA models autocorrelations in differenced data (stochastic trend). Exponential smoothing directly models level/trend/seasonality components (deterministic). ETS is more interpretable; ARIMA handles more complex autocorrelation structures.

3. **"How do you detect anomalies in a time series?"**
   - Fit a model and flag residuals beyond a threshold (e.g., > 3 std). Or use STL decomposition and check the remainder. For streaming: EWMA with control limits. For complex: Isolation Forest on time-windowed features.

4. **"Why can't you use random train/test splits for time series?"**
   - Creates data leakage: the model sees future data during training. Always split chronologically. Use expanding or sliding window cross-validation.
