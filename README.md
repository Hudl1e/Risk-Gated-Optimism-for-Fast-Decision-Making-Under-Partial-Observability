# Compressed POMDP + Risk-Gated Ensemble Q-Learning for Glucose Control

An RL-based artificial pancreas controller for Type 1 Diabetes, tested on the [SimGlucose](https://github.com/jxx123/simglucose) (UVA/Padova) simulator.

## Overview

The controller observes noisy CGM readings (a POMDP), learns a **pulse predictor** to forecast glucose dynamics, computes **action-conditional tail risk**, and selects insulin boluses via **risk-gated ensemble Q-learning**. A tiered safety shield prevents bolusing when glucose is low or not rising fast enough.

## Architecture

```
CGM_t → [ProxyFeaturizer] → o_t → [PulsePredictor] → p̂(a) → [pulse_risk] → ρ̂(a)
                                         ↓                         ↓
                                    [QEnsemble] ←── s̃ = [o_t, ρ̂(a)]
                                         ↓
                                  Qmix = (1-ρ̂)Q⁺ + ρ̂Q⁻
                                         ↓
                              [Safety Shield] → a_t (bolus)
```

## Where to Find Key Components

All code is in the main notebook. Here's a guide to each function/class:

### Observation & State

| Component | What it does |
|---|---|
| `ProxyFeaturizer` | Builds normalized observation `o_t = [CGM, ΔCGM, last_bolus, IOB]`. Tracks IOB via exponential decay (rate=0.98, ~103 min half-life). Maintains a sliding window of the last 8 observations for the pulse predictor. |
| `ProxyFeaturizer.o_vec()` | Called each step. Computes ΔCGM, updates IOB, normalizes features, appends to history window. |
| `ProxyFeaturizer.get_window()` | Returns the (8, 4) history matrix (zero-padded if < 8 steps), flattened as input to the pulse predictor. |

### Pulse Predictor (Learned Dynamics)

| Component | What it does |
|---|---|
| `PulsePredictor` | Small MLP (input: 33 = 8×4 + 1, hidden: 64, output: 1). Predicts `p̂ = ln(CGM_{t+1}/CGM_t)` given history window + candidate bolus. |
| `PulseTrainSample` | Dataclass storing one training sample: `(window_flat, bolus_norm, target_p)`. |
| Training target | `p* = ln(CGM_t / CGM_{t-1})` — the observed glucose log-return, clipped to [-0.5, 0.5]. Collected every step, trained every 4 steps via MSE. |

### Risk Estimation

| Component | What it does |
|---|---|
| `pulse_risk()` | **Core risk function.** Computes `ρ̂(a) = max(ρ_pulse, ρ_PK)` for each candidate action. |
| `ρ_pulse` | Pulse-based boundary risk: measures how much `CGM × e^{p̂}` violates the safe corridor [70, 180] beyond current violation. |
| `ρ_IOB` (sigmoid) | `(IOB+a)² / ((IOB+a)² + k²)` with k=3. CGM-independent — blocks stacking when total insulin is high regardless of sensor reading. |
| `ρ_margin` | `insulin_impact / (margin + insulin_impact)` where margin = CGM − 70. Higher risk when closer to hypo boundary. |
| `ρ_trend` | Additive penalty when CGM is falling (ΔCGM < −1) with IOB onboard. Captures active insulin still pushing BG down. |
| Zero-bolus risk | For `a=0`: uses proximity (how close to 70), momentum (30-min projected CGM), and IOB drag (remaining insulin impact). Provides learning signal even when no bolus is given. |

### Reward Function

| Component | What it does |
|---|---|
| `rl_shaped_reward()` | Returns shaped reward based on true BG. Target zone [100, 160] gives max reward (+4.0). Asymmetric: hypo penalties are 10-40× stronger than hyper penalties at equivalent distance from target. |
| Anti-stacking modifier | Penalizes bolusing when BG < 110: `reward -= 5.0 × bolus`. |
| Correction bonus | Rewards bolusing when BG > 180: up to `+15 × bolus` at BG > 250. Teaches the agent that insulin is the correct response to highs. |
| Surrogate reward | `r̃ = R_env - λ × ρ̂(a)` where λ=0.99. Used for Q-learning. Penalizes transitions through high-risk states. |

### Q-Learning (Ensemble TD)

| Component | What it does |
|---|---|
| `DiscreteQ` | Single Q-network. Takes state + action embedding, outputs scalar Q-value. Architecture: Linear(state+emb, 128) → ReLU → Linear(128, 128) → ReLU → Linear(128, 1). |
| `QEnsemble` | M=5 independent Q-networks. Provides `Q⁺ = max_m Q(m)` (optimistic) and `Q⁻ = min_m Q(m)` (pessimistic). |
| `Qmix` | Risk-gated interpolation: `Qmix(a) = (1-ρ̂)Q⁺ + ρ̂Q⁻`. Safe → optimistic. Risky → pessimistic. |
| Target network | Soft-updated: `Q̂ ← τQ + (1−τ)Q̂` with τ=0.005. Used for TD target computation. |
| `ReplayItem` | Stores `(s̃, a_idx, r̃, o_next, done)`. Buffer size = 20,000. |
| `_train_q_step()` | Samples batch of 128, computes TD target using target network (mean of Q⁺/Q⁻ for stability), updates all 5 critics via MSE. |

### Safety Shield & Action Selection

| Component | What it does |
|---|---|
| Hard shield | CGM < 110 → bolus = 0 always. |
| Soft shield | CGM 110–130 AND ΔCGM < 5 → bolus = 0 (no meal spike detected). |
| Moderate shield | CGM 130–150 AND ΔCGM < 2 → bolus = 0 (not rising). |
| Risk mask | Actions with `ρ̂(a) > ρ_max` (0.25) are excluded from selection. |
| ε-greedy | Among safe actions, random exploration with ε decaying from 0.3 → 0.02 over 5000 steps. |
| Teacher warmup | First 2000 steps: follow `MealBolusTeacher` (ICR-based meal boluses + clinical corrections). Collects pulse training data and RL replay during warmup. |

### Teacher (Warmup Policy)

| Component | What it does |
|---|---|
| `MealBolusTeacher` | Clinical basal-bolus policy. Gives meal bolus = grams/ICR when a meal is detected (±5 min window). Gives correction bolus = (CGM − target)/CF when CGM > 180 and no recent correction (2h cooldown). Capped at 3.0U. |

### Environment & Simulation

| Component | What it does |
|---|---|
| `_make_env_multiday()` | Creates a SimGlucose environment with multi-day meal scenarios. |
| `_gym_step()` | Wraps env.step() to handle both gym v0.15 (4-tuple) and v0.26+ (5-tuple) returns. |
| `_get_cgm()` / `get_true_bg()` | Extract CGM sensor reading and true plasma glucose from observation/info. |
| `_kovatchev_risk()` | Computes LBGI, HBGI, and total Risk Index from BG (standard clinical metric). |
| `bg_metrics()` / `daily_bg_metrics()` | Compute TIR, time below 70, mean BG — per-day and overall. |

## Key Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| Action set | (0, 0.2, 0.4, ..., 2.0) | Discrete bolus options in Units |
| IOB decay | 0.98/step | ~103 min half-life, matches rapid-acting insulin |
| Target zone | [90, 150] mg/dL | Wide zone prevents over-correction around 120–140 |
| Risk threshold (ρ_max) | 0.25 | Actions with >25% estimated violation probability are blocked |
| λ (risk penalty) | 0.99 | Weight of risk in surrogate reward |
| Warmup | 2000 steps (~100 hrs) | Teacher provides safe data for pulse model training |
| Ensemble size | 5 | Balances epistemic uncertainty estimation vs compute |
| Pulse window | 8 steps (24 min) | Captures recent meal/insulin dynamics |

## Running

```python
# Requires: pip install simglucose==0.2.1
# See the notebook for full simulation loop.

# Single patient run:
# 1. Define metadata (patient, meals, controller params)
# 2. Build MealBolusTeacher + CompressedRiskGatedRLController
# 3. Run simulation loop (train ~29 days, eval 14 days)
# 4. Compute metrics with daily_bg_metrics(df_eval)

# 10-patient replication:
# See replication_test.py cell at the bottom of the notebook
```

## File Structure

```
├── compressed_pomdp_pulse_rl.py    # Main algorithm (all classes + sim loop)
├── replication_test.py             # 10-patient batch evaluation (RL)
├── replication_test_pomdp.py       # 10-patient batch evaluation (pure POMDP baseline)
├── plot_final_2days.py             # 4-panel visualization of last 48h
├── debug_last_2days.py             # 3-min granularity debug table
└── algorithm_latex_final.tex       # Algorithm box for paper appendix
```
