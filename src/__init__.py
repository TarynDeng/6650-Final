"""Source package for the DSAN 6650 final project.

Modules:
    config       Every setting and its default (start here to change a decision).
    data         Price download, checksums, validation, features, date split.
    environment  The trading environment and its 9 actions.
    tabular_rl   State discretization, Q-learning, and SARSA.
    deep_rl      DQN with experience replay and a target network.
    training     The training loop shared by all algorithms.
    ensemble     Weighted and equal-weight voting over trained agents.
    evaluation   Metrics, policy rollouts, buy-and-hold, summary statistics.
    plots        Figures.
    runs         Run manifest and the checks that keep phases consistent.
    phases       Phase 1 (train and evaluate) and phase 2 (ensembles).
    report       Phase 3 (final tables and figures).
"""
