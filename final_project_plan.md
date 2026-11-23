# AI-Driven Smart Energy Management System (Smart EMS)
## Comprehensive Final Project Plan, Architecture, and Performance Evaluation

---

## 1. Executive Summary & Project Motivation

### 1.1 Problem Statement
Conventional Energy Management Systems (EMS) in microgrids rely heavily on **heuristic rule-based controllers** (e.g., greedy self-consumption). While simple to implement, rule-based systems are intrinsically myopic:
1. They discharge stored battery energy whenever a local generation deficit occurs, failing to anticipate future grid tariff spikes.
2. In the late afternoon (13:00–17:00), rule-based controllers deplete battery reserves on minor load deficits, leaving the battery starved of energy right as the expensive evening peak pricing window (17:00–22:00 at ₹0.30/kWh) begins.
3. They lack multi-horizon awareness of weather-dependent photovoltaic (PV) generation and fluctuating building demand patterns.

### 1.2 Proposed Solution
This project implements an **AI-Driven, Forecast-Assisted Energy Management System** that couples **Machine Learning (ML) short-term forecasting** with **Deep Reinforcement Learning (DRL)** continuous dispatch:
* **Prediction Layer (XGBoost):** Forecasts forward solar PV generation and campus electrical load demand across multi-hour rolling horizons.
* **Environment Layer (Gymnasium Microgrid):** Simulates microgrid physics (battery state-of-charge dynamics, charge/discharge efficiencies, degradation cost, and Time-of-Use tariffs) procedurally with zero external dataset dependencies.
* **Decision Layer (Forecast-Assisted PPO):** A continuous Actor-Critic Proximal Policy Optimization agent that ingests a 12-dimensional observation vector (combining real-time states and forward rolling forecasts) to modulate battery power dispatch in real time.

---

## 2. System Architecture & Methodology

```
┌────────────────────────────────────────────────────────────────────────┐
│                        DATA & SIMULATION LAYER                         │
│  - Procedural Mathematical Microgrid Simulation (Zero CSV dependency) │
│  - Gaussian Bell-Curve PV Generation (700 W base, 10% diurnal noise)   │
│  - Multi-Peak Campus Load Profile (Morning, Noon, Evening peaks)       │
│  - Dynamic Time-of-Use (ToU) Tariff Schedule (₹0.08 vs ₹0.30/kWh)      │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        ML FORECASTING SUBSYSTEM                        │
│  - Solar Generation Forecaster (XGBoost Regressor)                     │
│  - Load Demand Forecaster (XGBoost Regressor)                          │
│  - Rolling Multi-Horizon Feature Generator (15-min, 1-hour, 2-hour)    │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                  REINFORCEMENT LEARNING DISPATCH ENGINE                │
│  - 12-Dimensional State Space (Current State + Forward Predictions)    │
│  - Continuous Action Space: Battery Inverter Power [-1.5 kW, +1.5 kW]  │
│  - Algorithm: Forecast-Assisted PPO (Actor-Critic MLP)                 │
│  - Multi-Objective Economic & Safety Reward Function                   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                    BENCHMARKING & USER PRESENTATION                    │
│  - Fair Head-to-Head Comparison against Rule-Based Baseline EMS        │
│  - Evaluation Scopes: 1-Day (96 steps), 7-Day (672), 14-Day (1,344)    │
│  - Modern Glassmorphic Web Dashboard (FastAPI + Vanilla CSS/JS)        │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Working Plan & Implementation Phases

The project was executed across five sequential phases:

### Phase 1: Mathematical Microgrid Modeling & Physical Formulation
* Formulated the state equations for the Battery Energy Storage System (BESS):
  $$\text{SOC}_{t+1} = \text{SOC}_t + \frac{P_{\text{batt}, t} \cdot \eta \cdot \Delta t}{C_{\text{bat}}}$$
  where $\eta = \eta_{\text{ch}} = 0.95$ during charging and $\eta = \frac{1}{\eta_{\text{dis}}} = \frac{1}{0.95}$ during discharging.
* Established physical envelope constraints:
  $$0.20 \le \text{SOC}_t \le 0.90, \quad -1.5\text{ kW} \le P_{\text{batt}, t} \le +1.5\text{ kW}$$
* Modeled the Time-of-Use (ToU) tariff schedule:
  * **Off-Peak Rate:** ₹0.08/kWh ($00:00–17:00$ and $22:00–24:00$)
  * **Peak Rate:** ₹0.30/kWh ($17:00–22:00$, representing local utility peak demand hours)
* Designed procedural generation for solar irradiance and campus load profiles to eliminate reliance on static CSV files.

### Phase 2: Short-Term Forecasting Pipeline (XGBoost)
* Extracted temporal and lag features: diurnal sinusoidal harmonics ($\sin(\text{hour})$, $\cos(\text{hour})$), 15-min and 24-hour persistence lags, rolling means, and rolling standard deviations.
* Trained high-precision gradient boosted decision tree regressors (XGBoost) for forward solar and load predictions.
* Constructed the multi-horizon summary engine to output:
  * 15-minute immediate predictions
  * 1-hour forward rolling averages
  * 2-hour forward rolling averages and peak load projections

### Phase 3: Gymnasium Environment & Observation Space
* Developed `MicrogridEnv` and `SmartEnergyEnv` adhering to standard Gymnasium API (`reset()`, `step()`).
* Structured the **12-Dimensional Continuous Observation Vector**:
  $$\mathbf{s}_t = \begin{bmatrix}
  \hat{P}_{\text{load}, t} & \hat{P}_{\text{solar}, t} & \text{SOC}_t & \hat{C}_{\text{tariff}, t} \\
  \hat{P}_{\text{load}, t+1h} & \hat{P}_{\text{solar}, t+1h} & \hat{C}_{\text{tariff}, t+1h}^{\max} & \sin(2\pi t / 24) \\
  \cos(2\pi t / 24) & \hat{P}_{\text{load}, t+2h} & \hat{P}_{\text{solar}, t+2h} & \hat{C}_{\text{tariff}, t+2h}^{\max}
  \end{bmatrix}$$
  All features normalized into standard range $[0.0, 1.0]$.
* Formulated the multi-objective reward function:
  $$R_t = - \left( C_{\text{grid}, t} + w_{\text{deg}} |P_{\text{batt}, t}| + w_{\text{viol}} \cdot \mathbb{I}_{\text{SOC violation}} + w_{\text{curt}} \cdot P_{\text{curtail}} \right)$$

### Phase 4: Policy Optimization (PPO Training)
* Trained the Actor-Critic policy using Proximal Policy Optimization with clipped surrogate loss over 50,000 timesteps.
* Applied Generalized Advantage Estimation (GAE) for variance reduction.
* Verified policy convergence and stability across diverse random initial states and operational days.

### Phase 5: Head-to-Head Benchmarking & Interactive Dashboard
* Implemented identical evaluation environments to benchmark **Rule-Based EMS** vs. **Forecast-Assisted PPO**.
* Built a high-performance FastAPI backend (`/api/run/comparison`, `/api/results/comparison`) serving simulation data in <0.2 seconds.
* Designed a responsive, glassmorphic web dashboard providing real-time timeseries visualization and financial comparisons.

---

## 4. Models Used

### 4.1 Forecasting Models: XGBoost Regressors
Two independent Extreme Gradient Boosting (XGBoost) models provide continuous predictive awareness:
1. **Solar Generation Forecaster (`solar_xgb.pkl`):** Predicts next 15-minute solar power generation using historical generation lags and solar zenith angle proxies.
2. **Campus Load Demand Forecaster (`load_xgb.pkl`):** Predicts next 15-minute load demand using calendar features, tariff levels, and historical demand autocorrelations.

### 4.2 Reinforcement Learning Model: Forecast-Assisted PPO
* **Algorithm:** Proximal Policy Optimization (PPO), Actor-Critic framework.
* **Policy Architecture:** Multi-Layer Perceptron (MLP) with two hidden layers of 64 units each and Tanh activation functions.
* **Action Output:** Continuous battery power request $[-1.0, 1.0]$ scaled to battery inverter limits $[-1.5\text{ kW}, +1.5\text{ kW}]$.

### 4.3 Baseline Model: Conventional Rule-Based EMS
A greedy heuristic controller serving as the engineering reference benchmark:
* If Solar > Load: Charges battery with excess solar until $\text{SOC} = 0.90$; exports any remainder.
* If Load > Solar: Discharges battery to meet deficit until $\text{SOC} = 0.20$; draws shortfall from the grid.
* Ignores forward tariffs, upcoming solar patterns, and future peak windows.

---

## 5. Model Metrics & Hyperparameters

### 5.1 Forecasting Models Evaluation Metrics

Evaluated on 2,592 held-out out-of-sample test timesteps (27 full days):

| Metric | Solar PV Forecaster (XGBoost) | Campus Load Forecaster (XGBoost) |
| :--- | :--- | :--- |
| **Mean Absolute Error (MAE)** | **0.0287 kW** (28.7 W) | **0.2836 kW** |
| **Root Mean Squared Error (RMSE)** | **0.0617 kW** | **0.4574 kW** |
| **Coefficient of Determination ($R^2$)** | **0.9938** | **0.4563** |
| **Mean Absolute Percentage Error (MAPE)** | N/A (zero at night) | **29.82%** |
| **Test Dataset Samples** | 2,592 steps | 2,592 steps |
| **Primary Predictive Features** | • Current Solar (`solar_kw`: 75.4%)<br>• 24-hr Lag (`solar_lag_96`: 20.1%)<br>• Diurnal Clock (`cos_hour`: 2.5%) | • Tariff Level (`current_tariff`: 47.3%)<br>• Next-Hour Tariff (`next_h_avg`: 31.2%)<br>• Peak Tariff Indicator (`next_h_max`: 10.0%) |

### 5.2 Reinforcement Learning Model Hyperparameters & Specs

| Hyperparameter / Specification | Value | Technical Justification |
| :--- | :--- | :--- |
| **Algorithm** | PPO (Clip Objective) | Stable continuous policy updates without destructive policy collapse |
| **Policy Network Architecture** | MLP [64, 64] | Low parameter footprint, sub-millisecond inference time on edge devices |
| **Learning Rate ($\alpha$)** | $3 \times 10^{-4}$ | Standard Adam optimizer schedule for stable Actor-Critic convergence |
| **Discount Factor ($\gamma$)** | $0.99$ | Far-sighted horizon (~100 steps / 25 hours lookahead capability) |
| **GAE Parameter ($\lambda$)** | $0.95$ | Balances bias vs. variance in advantage estimation |
| **PPO Clip Range ($\epsilon$)** | $0.20$ | Constrains policy ratio $r_t(\theta) \in [0.8, 1.2]$ |
| **Entropy Coefficient ($c_2$)** | $0.005$ | Promotes exploration across charge/discharge thresholds during early training |
| **Rollout Buffer Size ($n_{\text{steps}}$)** | 2,048 | Captures multi-day operational trajectories per rollout batch |
| **Mini-Batch Size** | 64 | Efficient gradient descent batching |
| **Training Epochs per Update** | 10 | Optimizes sample efficiency across collected experience |
| **Total Training Steps** | 50,000 | Confirmed policy convergence and reward stabilization |
| **Observation Dimensions** | 12 | 4 real-time states + 6 forward forecast features + 2 cyclical temporal features |
| **Action Dimension** | 1 (Continuous) | Battery inverter power command $[-1.5\text{ kW}, +1.5\text{ kW}]$ |
| **SOC Safety Violation Rate** | **0.0%** (0 breaches) | Strict adherence to $20\% \le \text{SOC} \le 90\%$ battery physical limits |
| **Unserved Load Rate** | **0.0%** (0 kWh) | $100\%$ microgrid reliability across all benchmarked evaluation horizons |

---

## 6. Final Results Achieved

The table below summarizes the economic results achieved by **Forecast-Assisted PPO** compared against the **Rule-Based Baseline EMS** across all evaluation horizons.

> **Economic Metric Definitions:**
> * **Electricity Cost:** Direct expenditure paid to the grid utility for imported electrical energy over the evaluation period.
> * **Terminal-Adjusted Cost:** Total electricity cost adjusted for final stored battery energy against the initial state-of-charge ($C_{\text{adj}} = C_{\text{grid}} - \text{Credit}_{\text{final SOC}}$), ensuring a fair economic comparison that penalizes leaving the battery depleted.

### Benchmark Economic Performance

| Evaluation Horizon | Controller | Total Electricity Cost | Cost Optimization (%) | Terminal-Adjusted Cost | Adjusted Cost Advantage (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Single Operating Day**<br>*(96 timesteps / 24 hours)* | **Rule-Based EMS (Baseline)**<br>**Forecast-Assisted PPO** | ₹59.76<br>**₹34.86** | —<br>**41.7% Cost Reduction** | ₹60.82<br>**₹33.91** | —<br>**44.2% Economic Advantage** |
| **7-Day Benchmark**<br>*(672 timesteps / 168 hours)* | **Rule-Based EMS (Baseline)**<br>**Forecast-Assisted PPO** | ₹358.56<br>**₹214.97** | —<br>**40.0% Cost Reduction** | ₹370.18<br>**₹202.52** | —<br>**45.3% Economic Advantage** |
| **14-Day Benchmark**<br>*(1,344 timesteps / 336 hours)* | **Rule-Based EMS (Baseline)**<br>**Forecast-Assisted PPO** | ₹727.91<br>**₹420.81** | —<br>**42.2% Cost Reduction** | ₹740.36<br>**₹408.36** | —<br>**44.8% Economic Advantage** |

---

## 7. Conclusion

By integrating **XGBoost short-term solar and load forecasting** with a **continuous Proximal Policy Optimization (PPO)** controller, the Smart Energy Management System overcomes the core limitation of conventional rule-based systems. Across all benchmark horizons, the system delivers a consistent **~40% to 42.2% reduction in electricity costs** and a **~44% to 45.3% reduction in terminal-adjusted costs**, while maintaining complete physical safety and zero unserved load.
