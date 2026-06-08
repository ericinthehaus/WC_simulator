# run_analysis.py — your main analysis script

import logging
import pandas as pd
from scripts.aggregate_results import (
    run_monte_carlo,
    build_team_probabilities,
    build_r32_opponent_probabilities,
    build_goals_diagnostics
)
from scripts.third_place_scenarios import THIRD_PLACE_SCENARIOS_BY_GROUPS
import wc_simulator as wc

import json

logging.basicConfig(level=logging.INFO)

N = 100

matches_df, teams_df = wc.load_data()

placement_counts, r32_counts, all_goals = run_monte_carlo(
    matches_df, teams_df, N, THIRD_PLACE_SCENARIOS_BY_GROUPS
)

team_probs = build_team_probabilities(placement_counts, teams_df, N)
r32_probs  = build_r32_opponent_probabilities(r32_counts, teams_df, N)

completed = matches_df[matches_df['status'] == 'completed']
diagnostics = build_goals_diagnostics(all_goals, completed)

# Inspect
print(diagnostics['summary'])

r32_probs.to_csv('data_out/r32_probs.csv')
team_probs.to_csv('data_out/team_probs.csv')
diagnostics['simulated_dist'].to_csv('data_out/simulated_dis.csv')