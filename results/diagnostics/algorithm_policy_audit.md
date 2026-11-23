# RL Policy Differentiation Diagnostic Report

## 1. Model Loading

| Algorithm | Status | Size (bytes) | Param Fingerprint |
|-----------|--------|-------------|-------------------|
| PPO | OK | 154,393 | -0.22734111547470093 |
| SAC | OK | 3,112,027 | 0.8418501615524292 |
| TD3 | OK | 6,095,708 | 0.8131126761436462 |
| DQN | OK | 117,422 | -0.7280614376068115 |

## 2. Solar Priority Override

**32.3%** of timesteps were overridden by solar-priority logic (solar > load), bypassing the RL model.

> If this exceeds 40%, policy comparisons are partially masked.

## 3. Clipping Analysis

| Algorithm | Clipped Steps | % Clipped |
|-----------|--------------|-----------|
| PPO | 8 | 8.3% |
| SAC | 23 | 24.0% |
| TD3 | 30 | 31.2% |
| DQN | 10 | 10.4% |

## 4. Mean Absolute Action Differences

| Pair | MAD Raw Action | MAD Exec kW | % Same Executed (±0.05 kW) |
|------|---------------|-------------|----------------------------|
| ppo_vs_sac | 0.2946 | 0.4002 | 38.5% |
| ppo_vs_td3 | 0.3142 | 0.3609 | 46.9% |
| ppo_vs_dqn | 0.2346 | 0.2792 | 45.8% |
| sac_vs_td3 | 0.1416 | 0.2038 | 58.3% |
| sac_vs_dqn | 0.3021 | 0.4305 | 41.7% |
| td3_vs_dqn | 0.3253 | 0.4142 | 43.8% |

**Mean MAD exec (all pairs):** 0.3481 kW  |  **Mean % same:** 45.8%

## 5. Observation Consistency (Step 0)

- `ppo_vs_sac_obs0_maxdiff`: 0.000000
- `ppo_vs_td3_obs0_maxdiff`: 0.000000
- `ppo_vs_dqn_obs0_maxdiff`: 0.000000

## 6. DQN Action Mapping

DQN uses 11 discrete bins mapped to: `[np.float32(-1.0), np.float32(-0.8), np.float32(-0.6), np.float32(-0.4), np.float32(-0.2), np.float32(0.0), np.float32(0.2), np.float32(0.4), np.float32(0.6), np.float32(0.8), np.float32(1.0)]` in [-1.0, 1.0]

## 7. Controlled Single-State Tests

### A_high_tariff_high_soc
| Algorithm | Raw Action | Req kW | Exec kW | Clip Reason |
|-----------|-----------|--------|---------|-------------|
| PPO | 0.8836 | 1.5 | 1.5 | none |
| SAC | 0.5711 | 1.5 | 1.5 | none |
| TD3 | 0.719 | 1.5 | 1.5 | none |
| DQN | 0.2 | 0.9 | 0.9 | none |

### B_low_tariff_med_soc
| Algorithm | Raw Action | Req kW | Exec kW | Clip Reason |
|-----------|-----------|--------|---------|-------------|
| PPO | 0.2749 | 0.4123 | 0.4123 | none |
| SAC | -0.4882 | -0.7323 | -0.7323 | none |
| TD3 | -0.0888 | -0.1332 | -0.1332 | none |
| DQN | -0.2 | -0.3 | -0.3 | none |

### C_solar_surplus
| Algorithm | Raw Action | Req kW | Exec kW | Clip Reason |
|-----------|-----------|--------|---------|-------------|
| PPO | -0.6833 | 0.0 | 0.0 | idle_or_hightar |
| SAC | -0.9768 | 0.0 | 0.0 | idle_or_hightar |
| TD3 | -0.9806 | 0.0 | 0.0 | idle_or_hightar |
| DQN | -0.6 | 0.0 | 0.0 | idle_or_hightar |

### D_near_min_soc
| Algorithm | Raw Action | Req kW | Exec kW | Clip Reason |
|-----------|-----------|--------|---------|-------------|
| PPO | -0.38 | 0.0 | 0.0 | idle_or_hightar |
| SAC | -0.6659 | 0.0 | 0.0 | idle_or_hightar |
| TD3 | -0.1047 | 0.0 | 0.0 | idle_or_hightar |
| DQN | 0.2 | 0.228 | 0.228 | none |


---

## CONCLUSION

**Solar override:** 32.3%  |  **Mean MAD exec:** 0.3481 kW  |  **Mean % same:** 45.8%

> **4. GENUINE POLICY DIVERSITY -- Models produce clearly different actions; metric similarity driven by physical constraints.**

## Generated Artifacts

| File | Description |
|------|-------------|
| `model_action_comparison.csv` | Per-timestep actions for all 4 models |
| `action_difference_metrics.json` | Numeric MAD, clipping, obs metrics |
| `action_comparison.png` | Battery power trajectories |
| `soc_comparison.png` | SOC trajectories |
| `grid_comparison.png` | Grid import trajectories |
| `cumulative_cost_comparison.png` | Cumulative cost trajectories |