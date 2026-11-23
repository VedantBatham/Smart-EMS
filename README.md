# AI-Driven Smart Energy Management System Using Forecast-Assisted Proximal Policy Optimization

A production-grade, software-based final-year engineering system for sequential residential energy scheduling. The system coordinates energy flows between **Rooftop Solar PV**, **Battery Energy Storage System (BESS)**, the **Utility Grid**, and a dynamic **Household Electrical Load**.

The system compares a **Conventional Rule-Based EMS** baseline against the proposed **Forecast-Assisted PPO-Based EMS** under strictly identical operational profiles and Time-of-Use (ToU) tariffs.

---

## 1. Core Energy System Formulation

### 1.1 Physical Power Balance
At every 15-minute interval ($t$), household electrical demand must be continuously satisfied:

$$P_{\text{load}}(t) = P_{\text{solar\_used}}(t) + P_{\text{battery}}(t) + P_{\text{grid}}(t)$$

where the sign conventions are:
- $P_{\text{battery}} > 0$: Battery discharging (supplying power to load).
- $P_{\text{battery}} < 0$: Battery charging (absorbing power from excess solar or off-peak grid).
- $P_{\text{battery}} = 0$: Battery idle.
- $P_{\text{grid}} > 0$: Grid power import.

### 1.2 Battery Energy Storage System (BESS) Dynamics
Given nominal battery capacity $E_{\text{battery}} = 5.0\text{ kWh}$, time step $\Delta t = 0.25\text{ h}$ (15 minutes), charging efficiency $\eta_c = 0.95$, and discharging efficiency $\eta_d = 0.95$:

- **Charging State ($P_{\text{charge}} = |P_{\text{battery}}| > 0$ when $P_{\text{battery}} < 0$):**
  $$\text{SOC}_{t+1} = \text{SOC}_t + \frac{\eta_c \cdot P_{\text{charge}} \cdot \Delta t}{E_{\text{battery}}}$$

- **Discharging State ($P_{\text{discharge}} = P_{\text{battery}} > 0$):**
  $$\text{SOC}_{t+1} = \text{SOC}_t - \frac{P_{\text{discharge}} \cdot \Delta t}{\eta_d \cdot E_{\text{battery}}}$$

Physical operational boundaries:
$$\text{SOC}_{\min} = 0.20 \le \text{SOC}_t \le \text{SOC}_{\max} = 0.90$$
$$|P_{\text{battery}}| \le 1.5\text{ kW}$$

### 1.3 Time-of-Use (ToU) Electricity Tariff Schedule
Configured in `config.yaml` to simulate peak and off-peak cost structures:
- **00:00 – 06:00**: Rs. 3.00 / kWh *(Night Off-Peak)*
- **06:00 – 17:00**: Rs. 5.00 / kWh *(Standard Daytime)*
- **17:00 – 22:00**: Rs. 10.00 / kWh *(Evening Peak)*
- **22:00 – 24:00**: Rs. 4.00 / kWh *(Late Evening Off-Peak)*

---

## 2. Motivation: Rule-Based vs. Proposed PPO EMS

A conventional **Rule-Based EMS** operates reactively:
- When $P_{\text{solar}} \ge P_{\text{load}}$: solar supplies load; surplus charges the battery.
- When $P_{\text{solar}} < P_{\text{load}}$: battery discharges immediately until $\text{SOC}_{\min}$ is hit; grid meets remaining deficit.

**The Economic Flaw:** If the battery discharges during early afternoon (when electricity costs Rs. 5/kWh), the stored energy is exhausted before the Rs. 10/kWh evening peak begins. Consequently, the household must import expensive grid power at 19:00–21:00.

**The Proposed Forecast-Assisted PPO EMS:**
By receiving near-future predictions of solar generation, load demand, and scheduled tariff shifts, the PPO policy learns to **preserve** or strategically charge stored energy during cheap tariff hours, releasing it during expensive peak hours to maximize economic savings.

---

## 3. Machine Learning Forecasting Architecture

Short-term forecasters trained on 15-minute historical microgrid data using **XGBoost Regressors**:
- **Solar PV Forecaster (`models/solar_xgb.pkl`)**:
  - Predicts $\hat{P}_{\text{solar}, t+1}$ using lags (15m, 30m, 1h, 2h, 24h), rolling statistics (1h, 6h, 24h), and cyclical temporal encodings ($\sin/\cos$ hour and month).
  - Validation metrics: **$R^2 = 0.9938$, $\text{MAE} = 0.0287\text{ kW}$**.
- **Load Demand Forecaster (`models/load_xgb.pkl`)**:
  - Predicts $\hat{P}_{\text{load}, t+1}$ utilizing consumption lags, rolling means, and forward-looking tariff schedule features.
  - Validation metrics: **$\text{MAE} = 0.2836\text{ kW}$, $\text{MAPE} = 29.8\%$**.

---

## 4. Reinforcement Learning (PPO) MDP Formulation

- **Action Space:** Continuous $a_t \in [-1.0, 1.0]$:
  - $a = -1.0 \implies -1.5\text{ kW}$ (max charging)
  - $a = 0.0 \implies 0.0\text{ kW}$ (idle)
  - $a = +1.0 \implies +1.5\text{ kW}$ (max discharging)
- **Observation Space (12 dimensions):**
  1. Normalized current load ($P_{\text{load}} / 5.0$)
  2. Normalized current solar ($P_{\text{solar}} / 4.0$)
  3. Current Battery $\text{SOC}$ $[0.2, 0.9]$
  4. Normalized current tariff ($\text{Tariff} / 12.0$)
  5. Next 1-hour average load forecast ($\hat{P}_{\text{load}, 1\text{h}}$)
  6. Next 1-hour average solar forecast ($\hat{P}_{\text{solar}, 1\text{h}}$)
  7. Next 1-hour maximum tariff ($\text{Tariff}_{1\text{h}, \max}$)
  8. Hour cyclical $\sin(\theta)$
  9. Hour cyclical $\cos(\theta)$
  10. Next 2-hour average load forecast ($\hat{P}_{\text{load}, 2\text{h}}$)
  11. Next 2-hour average solar forecast ($\hat{P}_{\text{solar}, 2\text{h}}$)
  12. Next 2-hour maximum tariff ($\text{Tariff}_{2\text{h}, \max}$)
- **Reward Function:**
  $$R_t = -\Big[ C_{\text{grid}, t} + \lambda_1 C_{\text{battery}, t} + \lambda_2 P_{\text{soc}, t} + \lambda_3 P_{\text{unserved}, t} \Big] - \lambda_{\text{terminal}} |\text{SOC}_{\text{final}} - \text{SOC}_{\text{target}}|^2$$
  - $C_{\text{grid}, t} = P_{\text{grid}, t} \cdot \text{Tariff}_t \cdot 0.25\text{ h}$
  - Battery degradation penalty: $\lambda_1 |P_{\text{battery}}| \cdot 0.25$
  - Terminal SOC alignment ensures fair economic comparison without the agent artificially emptying the battery.

---

## 5. Hybrid Data + PPO Training Strategy

To ensure the PPO policy develops robust, generalizable decision rules and is **not trained only on a fixed historical CSV trajectory**, the system adopts a hybrid training framework:

> **Historical solar-generation and load-demand profiles are used as realistic base trajectories. During PPO training, controlled stochastic variations are introduced in renewable generation, load demand, and initial battery SOC to create diverse scenarios and reduce memorization of fixed historical sequences. The trained controller is finally evaluated on unseen, unmodified historical data and compared with a conventional rule-based EMS under identical operating conditions.**

### 5.1 Chronological Dataset Partitioning
The historical Open Power System Data (OPSD) household dataset is segregated chronologically:
- **70% Training Partition** (`train_data.csv`, 126 days): Base trajectory pool for scenario randomization.
- **15% Validation Partition** (`validation_data.csv`, 27 days): Hyperparameter tuning & model check.
- **15% Final Unseen Test Partition** (`test_data.csv`, 27 days, June 2020): **Strictly untouched** during PPO training and tuning.

### 5.2 Stochastic Perturbation Dynamics
For each PPO episode, a real base historical day is sampled and perturbed according to:
$$P^{\text{train}}_{\text{solar}}(t) = P^{\text{base}}_{\text{solar}}(t) \cdot k_s + \epsilon_s(t), \quad k_s \sim U(0.7, 1.1)$$
$$P^{\text{train}}_{\text{load}}(t) = P^{\text{base}}_{\text{load}}(t) \cdot k_l + \epsilon_l(t), \quad k_l \sim U(0.8, 1.2)$$

- **Realistic Solar Weather Events:**
  - Clear sunny days
  - Correlated midday cloud dips (30–70% drop between 12:00 and 14:00)
  - Morning cloud cover suppression
  - Afternoon convective cloud dips
  - Strict physical adherence: $P_{\text{solar}} \ge 0$, and nighttime solar ($h < 6$ or $h \ge 19$) is identically zero.
- **Realistic Household Load Dynamics:**
  - Morning and evening peak scaling
  - Peak timing shifts ($\pm 15$ to $\pm 30$ minutes)
  - Short-duration high-power appliance demand spikes
- **Randomized Battery State-of-Charge Initialization:**
  $$\text{SOC}_0 \sim U(0.30, 0.80)$$
  Forces the policy to optimize decisions from diverse initial battery states rather than assuming a single static starting level.
- **Forward Forecast Feature Uncertainty:**
  Derived directly from perturbed profiles with realistic residual noise ($\sigma_s = 0.05, \sigma_l = 0.08$).

### 5.3 Unbiased Final Evaluation Protocol
To guarantee strict scientific validity:
- **Zero random perturbations** are applied to the primary benchmark.
- Both the Rule-Based EMS and the Forecast-Assisted PPO controller execute on identical, untouched historical test trajectories with matched battery hardware parameters, initial SOC, and tariff rates.
- Optional robustness evaluations are conducted across segregated adversarial stress scenarios (high load, low solar, midday cloud drop, evening surge).

---

## 6. Project Directory Structure

```
AI_Smart_Energy_Management/
├── README.md
├── requirements.txt
├── config.yaml
├── .gitignore
├── data/
│   ├── raw/household_data.csv
│   ├── processed/
│   │   ├── energy_data.csv
│   │   ├── train_data.csv
│   │   ├── validation_data.csv
│   │   └── test_data.csv
│   └── results/
│       ├── rule_based_results.csv
│       ├── ppo_results.csv
│       └── comparison_results.csv
├── preprocessing/
│   ├── load_data.py
│   ├── preprocess.py
│   └── feature_engineering.py
├── forecasting/
│   ├── train_solar_model.py
│   ├── train_load_model.py
│   ├── forecast.py
│   └── evaluate_forecast.py
├── energy_system/
│   ├── battery.py
│   ├── tariff.py
│   ├── power_balance.py
│   └── simulator.py
├── controllers/
│   ├── rule_based.py
│   └── controller_utils.py
├── environment/
│   └── smart_energy_env.py
├── rl/
│   ├── train_ppo.py
│   ├── evaluate_ppo.py
│   ├── ppo_service.py
│   └── callbacks.py
├── models/
│   ├── solar_xgb.pkl
│   ├── load_xgb.pkl
│   └── forecast_ppo_ems.zip
├── evaluation/
│   ├── metrics.py
│   ├── compare_controllers.py
│   └── generate_plots.py
├── backend/
│   ├── main.py
│   ├── routes/
│   │   ├── system.py
│   │   ├── forecast.py
│   │   ├── simulation.py
│   │   └── comparison.py
│   ├── schemas/simulation_schema.py
│   └── services/
│       ├── simulation_service.py
│       ├── forecast_service.py
│       └── comparison_service.py
├── frontend/
│   ├── index.html
│   ├── css/style.css
│   └── js/
│       ├── api.js
│       ├── charts.js
│       ├── simulation.js
│       └── dashboard.js
├── results/
│   ├── plots/
│   ├── metrics/
│   └── training/
├── tests/
│   ├── test_battery.py
│   ├── test_tariff.py
│   ├── test_power_balance.py
│   ├── test_environment.py
│   ├── test_controllers.py
│   └── test_api.py
└── scripts/
    ├── run_preprocessing.py
    ├── train_forecasting.py
    ├── train_ppo.py
    ├── run_comparison.py
    └── run_app.py
```

---

## 6. Installation & Execution Guide

### 6.1 Install Dependencies
```bash
pip install -r requirements.txt
```

### 6.2 Run Preprocessing Pipeline
```bash
python scripts/run_preprocessing.py
```

### 6.3 Train XGBoost Forecasting Models
```bash
python scripts/train_forecasting.py
```

### 6.4 Train PPO Reinforcement Learning Policy
```bash
python scripts/train_ppo.py
```

### 6.5 Run Head-to-Head Comparative Simulation
```bash
python scripts/run_comparison.py
```

### 6.6 Run Automated Test Suite
```bash
python -m pytest tests/
```

### 6.7 Launch Web Application Dashboard
```bash
python scripts/run_app.py
```
Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in any modern web browser to interact with the dashboard.
