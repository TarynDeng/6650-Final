# Project Proposal

**Comparative Study of Shallow and Deep Reinforcement Learning Algorithms in Stock Trading with Performance-Based Weighted Ensemble**

DSAN 6650 (Reinforcement Learning), Final Project

> **Status: draft.** This document is an English translation and reorganization of the team's original project outline. Items marked **(tentative)** are not final. Equations marked **(reconstructed)** were lost when the outline was exported to plain text; they have been rewritten in standard notation from the surrounding text, and **the team should confirm them**. Content marked *added during repo setup* is not in the original outline. The full list of open questions is in [Section 12](#12-open-decisions).

> **Plan vs. implementation.** This document is the original plan and is kept as written. Where the implementation made a concrete choice, a note marked **Implemented** says what was built. The current defaults and their status are in the [README, Section 8](../README.md#8-decisions-and-their-status).
>
> | Plan | Implemented (placeholder defaults in `src/config.py`) |
> |---|---|
> | Existing trading environment such as Gym Trading Env | Small custom Gymnasium environment (`src/environment.py`), tested |
> | About 3 stocks, 8–10 years | AAPL, JPM, XOM; Yahoo Finance adjusted prices, 2015–2025 (11 years), committed with checksums |
> | Validation for limited hyperparameter selection and ensemble weights | Ensemble weights only; no hyperparameter tuning; every agent uses its final model |
> | Cash and position fractions in the state | Position fraction only (cash fraction = 1 − position fraction) |
> | Optional DQN-on-discrete-states control | Not run |
> | Reward $r_t = \ln(V_t/V_{t-1})$ | Same reward, indexed as $r_t = \ln(V_{t+1}/V_t)$: the order fills at day $t$'s close and is valued at day $t+1$'s close |
> | Discount factor, temperature τ, tie-break: open | γ = 0.99; temperature 0.05; ties → top-weight member, then Hold, then lowest action |

---

## Contents

1. [Project Title](#1-project-title)
2. [Project Overview](#2-project-overview)
3. [Research Questions](#3-research-questions)
4. [RL Algorithms](#4-rl-algorithms)
5. [Trading Environment](#5-trading-environment)
6. [Experimental Design](#6-experimental-design)
7. [Data Splitting](#7-data-splitting)
8. [Evaluation Metrics](#8-evaluation-metrics)
9. [Scope Control](#9-scope-control)
10. [Deliverables](#10-deliverables)
11. [Implementation Plan](#11-implementation-plan)
12. [Open Decisions](#12-open-decisions)
13. [Notation Summary](#13-notation-summary)

---

## 1. Project Title

Comparative Study of Shallow and Deep Reinforcement Learning Algorithms in Stock Trading with Performance-Based Weighted Ensemble

## 2. Project Overview

The project uses a historical stock-trading simulation as a common experimental environment to systematically compare **shallow (tabular) RL** and **deep RL** algorithms. The focus is on the RL algorithms themselves, not on building a sophisticated trading agent.

Given the market and account state, each algorithm learns on its own:

- when to buy;
- when to sell;
- how much to buy or sell each time;
- when to keep the current position.

After training, the algorithms are compared on a held-out historical test period in terms of profitability, terminal loss, learning efficiency, and stability. The project also studies whether a **weighted vote over several RL policies**, with weights based on validation-set performance, can outperform the individual algorithms.

All experiments use historical market data and simulated capital. **No real trading is involved.**

## 3. Research Questions

| ID | Theme | Question |
|----|-------|----------|
| **RQ1** | Algorithm Performance | In the same stock-market environment, how does the trading performance of shallow RL differ from that of deep RL? |
| **RQ2** | Terminal Loss Distribution | How do the algorithms differ in the frequency and severity of terminal investment losses? |
| **RQ3** | Learning Efficiency and Stability | How do the algorithms differ in learning efficiency, training stability, and results across random seeds? |
| **RQ4** | Weighted Ensemble | Can a weighted combination built from validation-set performance improve on the individual algorithms in return, terminal-loss outcomes, and stability? |

## 4. RL Algorithms

| Category | Algorithm | Characteristics |
|----------|-----------|-----------------|
| Shallow RL | Q-learning | Tabular, off-policy |
| Shallow RL | SARSA | Tabular, on-policy |
| Deep RL | DQN | Neural network, off-policy |

All three algorithms are covered in the course and homework: HW-1 provides the basis for tabular RL, and HW-2 provides the basis for DQN, PyTorch, and Gymnasium/RLlib.

DQN uses the basic structure taught in the course, including **experience replay** and a **target network**. Additional, more complex algorithms (PPO, SAC, Rainbow DQN, Meta-RL, etc.) are out of scope for now.

## 5. Trading Environment

### 5.1 Environment Choice

The first preference is to evaluate an existing, Gymnasium-compatible historical stock-trading environment (for example, *Gym Trading Env*), and to avoid building a complex trading system from scratch.

> **Implemented:** a small custom Gymnasium environment. Gym Trading Env's actions are target positions, not "buy x% of cash / sell x% of holdings" orders, and a short custom environment is easier to verify.

The environment must support:

- reading historical prices, volume, and related data;
- executing buy, sell, and hold actions;
- updating the cash balance, stock position, and portfolio value;
- computing transaction costs;
- returning observation, reward, termination, and related information to the algorithm.

**To verify:** whether the chosen environment directly supports the action design below (orders sized as a fraction of available cash or current holdings). If not, a simple action-adapter layer is needed, and the fill prices and portfolio accounting must be checked for correctness.

### 5.2 Stock Data

- **Data length (tentative):** about 8–10 years of daily historical prices. *Implemented: 2015–2025, i.e. 11 years.*
- **Number of stocks (tentative):** about 3, chosen to cover different market behaviors or sectors where possible.
- **Simulation:** each episode trades a single stock; experiments are repeated across stocks and test windows.
- **Trading scope:** no multi-stock portfolio allocation in the initial version.

The specific stocks, data source, and years are **not yet final**.

### 5.3 State Space

**Market information**

- recent stock return (e.g., 5-day return);
- longer-window return or price trend (e.g., 20-day return);
- historical volatility (e.g., 20-day volatility).

**Portfolio information**

- current cash fraction;
- current position (holdings) fraction.

Tabular Q-learning and SARSA use a **discretized** state. DQN may use **continuous** numerical features derived from the same information. **Discretization boundaries and normalization parameters must be fit on the training set only.**

**Optional control experiment:** additionally train DQN on the same discrete state encoding used by the tabular methods, to test whether the state representation changes the conclusions of the comparison.

> **Implemented:** market features 5-day and 20-day log return and 20-day volatility; the portfolio feature is the position fraction only, because the cash fraction is 1 minus it. Tabular agents use 3 quantile bins per market feature (fit on training data) × 5 position levels = 135 states; DQN uses the four standardized numbers. The optional control experiment was not run.

### 5.4 Action Space

So that the algorithms learn timing, direction, and size together, while staying compatible with the discrete-action setting of Q-learning, SARSA, and DQN, the initial design uses **Discrete Trade Sizing with 9 actions (tentative)**:

| Action | Operation |
|:------:|-----------|
| 0 | Hold (no trade) |
| 1 | Buy with 25% of available cash |
| 2 | Buy with 50% of available cash |
| 3 | Buy with 75% of available cash |
| 4 | Buy with 100% of available cash |
| 5 | Sell 25% of current holdings |
| 6 | Sell 50% of current holdings |
| 7 | Sell 75% of current holdings |
| 8 | Sell 100% of current holdings |

These percentages are **selectable trade-size levels**, not a predefined trading strategy or risk preference. The algorithm decides which action to take at each step.

The environment computes the actual number of shares filled from the current price, cash, holdings, and fees. The initial rules are **no leverage and no short selling**. Still to be agreed: whether fractional shares are allowed, how share counts are rounded to integers, and how to handle an order when cash is insufficient to cover fees.

> **Implemented:** fractional shares; a 0.1% fee on the traded value; orders fill at the day's close. A buy spends the chosen fraction of cash *including* the fee, so cash never goes negative; trades worth under one cent are skipped.

### 5.5 Reward Function

The initial reward is the **per-step net log return of the portfolio value** **(reconstructed)**:

```math
r_t = \ln\left(\frac{V_t}{V_{t-1}}\right)
```

where $V_t$ is the portfolio value (cash plus the market value of holdings) at time $t$, and transaction costs are already reflected in the change in portfolio value. The formula requires $V_t > 0$ for all $t$ **(reconstructed)**.

- **Training reward:** the per-step feedback used for RL learning.
- **Evaluation metrics:** the terminal investment return, terminal loss, and stability measures computed after the test period ends.

The two are related but not the same. With discount factor $\gamma = 1$, the cumulative log return telescopes to $\sum_{t=1}^{T} r_t = \ln(V_T / V_0)$, which is a monotonic function of final portfolio growth; with $\gamma < 1$, the discounted return is no longer strictly equivalent **(reconstructed: the two cases were blank in the export and are assumed to be $\gamma = 1$ versus $\gamma < 1$)**. The training discount factor must be the same for all algorithms and stated in the report.

> **Implemented:** the code writes the same reward as $r_t = \ln(V_{t+1}/V_t)$ (order at day $t$'s close, valuation at day $t+1$'s close), and uses γ = 0.99 for every algorithm, i.e. the "not strictly equivalent" case above.

No maximum-drawdown penalty, terminal-loss penalty, risk-aversion coefficient, or hand-crafted stop-loss rule is added.

## 6. Experimental Design

### Phase 1: Individual Algorithm Comparison

Train Q-learning, SARSA, and DQN separately, making sure they share the same:

- stocks and historical data split;
- legal actions and transaction costs;
- initial capital and test windows;
- reward definition;
- training and evaluation procedure, with random seeds recorded.

The main comparisons are terminal return, the terminal-loss distribution, learning curves, training stability, and the distribution of trading actions each algorithm selects.

Where possible, the training budget is expressed in **environment steps**. Since the algorithms have different computational costs, wall-clock training time can be reported separately.

### Phase 2: Performance-Based Weighted Ensemble

Using the three trained policies, measure their performance on the **validation set** and use it to fix the ensemble weights.

One simple scheme is a **softmax over the mean validation terminal return** **(reconstructed)**. *Implemented: with one validation year, each member's "mean" is its single 2022 episode for that stock and seed; τ is fixed at 0.05 (called `temperature` in the code). In the runs, one member always got more than half the weight, so the weighted vote followed that member.*

```math
w_i = \frac{\exp\left(\bar{R}_i / \tau\right)}{\sum_{j=1}^{N} \exp\left(\bar{R}_j / \tau\right)}, \qquad i = 1, \dots, N
```

where $\bar{R}_i$ is the mean validation terminal return of algorithm $i$, $N$ is the number of policies ($N = 3$ here), and $\tau > 0$ is a temperature parameter that is either fixed in advance or selected using the validation set only.

At each trading step, the actions recommended by the policies are combined by **weighted voting** **(reconstructed)**:

```math
S_t(a) = \sum_{i=1}^{N} w_i \, \mathbf{1}\left[\pi_i(s_t) = a\right]
```

```math
a_t^{\text{ens}} = \arg\max_{a \in \mathcal{A}} S_t(a)
```

where $\pi_i(s_t)$ is the greedy action of policy $i$ in state $s_t$, $\mathbf{1}[\cdot]$ is the indicator function, and $\mathcal{A}$ is the action set {0, 1, …, 8}.

Rules:

- The test set is **never** used to tune the weights.
- An **equal-weight ensemble** ($w_i = 1/N$) is included as a control.
- Each policy in the ensemble must recompute its action from the **ensemble's current account state** at every step, rather than splicing together actions from each policy's own independent backtest.
- If votes tie, a fixed, reproducible **tie-breaking rule** must be specified in advance.

### Phase 3: Out-of-Sample Evaluation

The final comparison includes:

| Method | Type |
|--------|------|
| Q-learning | Shallow RL |
| SARSA | Shallow RL |
| DQN | Deep RL |
| Performance-Based Weighted Ensemble | Policy combination |
| Equal-Weight Ensemble | Ensemble baseline |
| Buy-and-Hold | Non-RL benchmark |

Every method runs a historical trading simulation over the held-out test windows, and the full portfolio-value trajectory and the final results are recorded.

## 7. Data Splitting

Data are split **chronologically** to prevent data leakage. Tentative plan:

| Split | Period (tentative) | Purpose |
|-------|--------------------|---------|
| Training | 2015–2021 | Train the RL algorithms |
| Validation | 2022 | Select a limited set of hyperparameters; set the ensemble weights *(implemented: ensemble weights only; no tuning)* |
| Testing | 2023–2025 | Final out-of-sample comparison |

The test period can be split into three fixed 12-month windows, e.g., 2023, 2024, and 2025. The 12-month length is a common evaluation horizon, not a research variable.

Each RL algorithm should be trained with **multiple random seeds (e.g., 5, tentative)** to measure the effect of training randomness on the results.

Principles:

- Do not retrain or select parameters on the test set.
- Normalization statistics, discretization thresholds, and similar quantities are fit on training data only.
- Define clearly when market data are observed and when orders are executed, to avoid look-ahead bias.
- Do not treat test results from different random seeds on the same market window as fully independent market samples.

## 8. Evaluation Metrics

### 8.1 Terminal Return

**(reconstructed)**

```math
R_T = \frac{V_T - V_0}{V_0}
```

where $V_0$ is the initial capital and $V_T$ is the portfolio value at the end of the test window. Report the mean and median terminal return, and results per market window.

### 8.2 Terminal Loss

**(reconstructed)**

```math
L_T = \max\left(0, \; -R_T\right) = \max\left(0, \; \frac{V_0 - V_T}{V_0}\right)
```

Each test episode is classified by its final outcome:

| Category | Final investment outcome |
|----------|--------------------------|
| Category 1 | Profit or break-even ($L_T = 0$) |
| Category 2 | Loss > 0% and ≤ 20% |
| Category 3 | Loss > 20% and ≤ 40% |
| Category 4 | Loss > 40% |

> **Important: terminal loss is not maximum drawdown.** Only the loss at the end of the test window, relative to initial capital, is counted. The peak-to-trough maximum drawdown during the episode is **not** used as a substitute.

If some loss categories do not occur in the test data, report a frequency of zero honestly, and also show the continuous return distribution. *(Implemented so far: category shares and per-episode returns in `results/tables/phase3_test.csv`; a distribution plot is not produced yet.)* Do not construct artificial severe-loss cases.

### 8.3 Learning Efficiency

- Training / evaluation return vs. environment steps.
- Performance within a given sample budget.
- Whether the learning curves stabilize.

Do not claim, without evidence, that any algorithm has converged to the theoretically optimal policy in the stock-trading environment.

### 8.4 Stability

- Mean and standard deviation of results across random seeds.
- Performance differences across stocks and test years.
- Stability of the ensembles compared with the individual algorithms.

### 8.5 Trading Behavior (supporting analysis)

- Buy / sell / hold frequencies.
- Selection frequency of each trade-size action.
- Number of trades and total transaction costs.
- Portfolio-value curves.

## 9. Scope Control

The project uses techniques already covered in the course and homework where possible:

- MDPs, the Bellman equation, temporal-difference learning;
- Q-learning, SARSA, ε-greedy exploration;
- DQN, experience replay, target network;
- PyTorch, Gymnasium, and RLlib if needed;
- basic statistics, plotting and analysis, and controlled experimental design.

The step up from the homework is mainly: a common benchmark on real historical market data, a systematic shallow-vs-deep RL comparison, multiple seeds with out-of-sample validation, the terminal-loss distribution, and the performance-based weighted ensemble.

**Not included for now:** continuous-action actor-critic methods, large neural networks, multi-asset portfolio optimization, Meta-RL, complex risk-adjusted rewards, or a real market execution system.

## 10. Deliverables

Based on the course project requirements as currently understood:

1. **Original code:** reproducible training, testing, ensemble, and evaluation code.
2. **Research paper (4–6 pages):** research questions, algorithms, experimental design, results, limitations, and conclusions.
3. **Presentation (10–15 minutes):** methods and main experimental findings. **Due 12/2.**
4. **Private GitHub repository:** developed in the repository with its commit history kept.

Suggested code layout (the original plan; the actual layout is in the [README, Section 5](../README.md#5-repository-layout)):

```
6650-Final/
├── data/
├── src/
│   ├── environment.py
│   ├── tabular_rl.py
│   ├── deep_rl.py
│   ├── ensemble.py
│   └── evaluation.py
├── experiments/
│   └── run_experiments.py
├── results/
├── requirements.txt
└── README.md
```

## 11. Implementation Plan

1. **Minimal working experiment:** confirm the historical data, trading environment, action execution, cash/holdings accounting, and reward are correct; get all three algorithms running on one stock first.
2. **Individual algorithm training and comparison:** unify the training procedure, save models and learning curves, and complete validation.
3. **Weighted ensemble:** set the weights from validation results only; implement weighted voting and the equal-weight baseline.
4. **Scale up testing:** add more stocks, fixed yearly test windows, and multiple random seeds.
5. **Analysis and reporting:** summarize terminal return, terminal-loss categories, learning efficiency, and stability; produce figures, the paper, and the presentation.
6. **Optional extension:** if the core experiments are done, run the control experiment with DQN on the discrete state encoding.

## 12. Open Decisions

Each item now has a placeholder default in `src/config.py`; the current values and whether the team has confirmed them are tracked in the [README, Section 8](../README.md#8-decisions-and-their-status). The original list:

- Historical data source, final stock list, and date range. *Default: yfinance; AAPL, JPM, XOM; 2015–2025.*
- Choice of trading environment. *Default: custom Gymnasium environment.*
- Whether to keep the 9 actions; integer vs. fractional shares; commission and execution-price rules. *Default: 9 actions, fractional shares, 0.1% fee, fill at the day's close.*
- State features, discretization levels, and the size of the tabular state space. *Default: 135 states.*
- Whether DQN uses the HW-2 PyTorch implementation or RLlib. *Default: HW-2-style PyTorch.*
- Training budget, number of random seeds, and hyperparameter search ranges. *Default: 100k steps, 5 seeds, no search.*
- Weighted-ensemble temperature τ and the vote tie-breaking rule. *Default: 0.05; top-weight member, then Hold, then lowest action.*
- Team roles and the GitHub workflow. *Open.*
- *(added during repo setup)* Paper template and slide software. *Default: LaTeX `article`; PowerPoint deck in `slides/`.*
- *(added during repo setup)* Discount factor γ. *Default: 0.99.*
- *(added during repo setup)* Repository visibility. *Open: the repository is public; the course asks for a private one.*
- *(added during repo setup)* Confirm every equation marked **(reconstructed)** in this document. *Open.*

## 13. Notation Summary

*Added during repo setup; not part of the original outline.*

| Symbol | Meaning |
|--------|---------|
| $t$ | Trading step (one trading day) |
| $V_t$ | Portfolio value at step $t$ (cash + market value of holdings, after fees) |
| $V_0$, $V_T$ | Initial capital; portfolio value at the end of the episode |
| $r_t$ | Per-step reward, $\ln(V_t / V_{t-1})$ |
| $\gamma$ | Discount factor, shared across all algorithms |
| $R_T$ | Terminal return, $(V_T - V_0)/V_0$ |
| $L_T$ | Terminal loss, $\max(0, -R_T)$ |
| $\bar{R}_i$ | Mean validation terminal return of policy $i$ |
| $\tau$ | Softmax temperature for the ensemble weights |
| $w_i$ | Ensemble weight of policy $i$ |
| $\pi_i(s_t)$ | Greedy action of policy $i$ in state $s_t$ |
| $\mathcal{A}$ | Action set {0, 1, …, 8} |

---

*Summary of the project's positioning:* under a common historical market simulation, compare how the shallow and deep RL algorithms covered in the course learn trade timing and trade size on their own, and test whether a performance-based policy ensemble gives a reproducible improvement.
