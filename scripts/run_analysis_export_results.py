"""
run_analysis_export_results.py

Orchestration script — runs the Monte Carlo simulation and exports
all output JSON files for the dashboard.

Imports all aggregation logic from aggregate_results.py.
This file contains only orchestration, export, and payload-building.

Usage:
    python scripts/run_analysis_export_results.py

Output files (written to docs/output/):
    probabilities.json      — group stage placement + advancement odds
    r32_opponents.json      — likely Round of 32 opponents per team
    match_outcomes.json     — W/D/L probabilities per group stage match
    conditional_outcomes.json — placement odds conditioned on matchday-3 result
"""

import json
import logging
import os
from datetime import date

import wc_simulator as wc
from constants import MEAN_GOALS, N_SIMULATIONS, DB_PATH
from aggregate_results import (
    run_monte_carlo,
    build_team_probabilities,
    build_r32_opponent_probabilities,
    build_match_outcomes,
    build_conditional_probabilities,
    build_goals_diagnostics,
)

# ============================================================
# CONFIG
# ============================================================

OUTPUT_DIR       = 'docs/output'
SHOW_CONDITIONAL = True   # set False to hide the conditional tab on dashboard

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


# ============================================================
# SECTION 1 — PAYLOAD BUILDERS
# (thin wrappers that add meta + export shape on top of
#  the DataFrames returned by aggregate_results.py)
# ============================================================

def build_probabilities_payload(team_probs_df, matches_df, n_simulations):
    """
    Wrap the team probabilities DataFrame into the JSON payload shape
    expected by the dashboard, adding the meta block.

    Args:
        team_probs_df (pd.DataFrame): Output of build_team_probabilities().
        matches_df (pd.DataFrame): Full matches table — used for meta counts.
        n_simulations (int): Number of simulation runs.

    Returns:
        dict: Ready to pass to export_json().
    """
    group_matches     = matches_df[matches_df['stage_id'] == 1]
    matches_completed = int((group_matches['status'] == 'completed').sum())
    matches_total     = int(len(group_matches))

    return {
        'meta': {
            'last_updated':      str(date.today()),
            'n_simulations':     n_simulations,
            'mean_goals':        MEAN_GOALS,
            'matches_completed': matches_completed,
            'matches_total':     matches_total,
        },
        'teams': team_probs_df.to_dict(orient='records'),
    }


def build_r32_payload(r32_probs_df, n_simulations):
    """
    Wrap the R32 opponent probabilities DataFrame into the JSON payload
    shape expected by the dashboard.

    The dashboard expects a list of team objects each containing a nested
    'opponents' list. This function reshapes the flat DataFrame accordingly.

    Args:
        r32_probs_df (pd.DataFrame): Output of build_r32_opponent_probabilities().
        n_simulations (int): Number of simulation runs.

    Returns:
        dict: Ready to pass to export_json().
    """
    matchups_out = []

    for (team_id, team_name, group, crest_url), group_df in r32_probs_df.groupby(
        ['team_id', 'team_name', 'group', 'crest_url'], dropna=False
    ):
        opponents = group_df.sort_values('probability', ascending=False)[[
            'opponent_id', 'opponent_name', 'opponent_group', 'probability'
        ]].to_dict(orient='records')

        matchups_out.append({
            'team_id':   int(team_id),
            'team_name': team_name,
            'group':     group,
            'crest_url': crest_url if crest_url and str(crest_url) != 'nan' else None,
            'opponents': opponents,
        })

    matchups_out.sort(key=lambda t: t['team_name'])

    return {
        'meta': {
            'last_updated':  str(date.today()),
            'n_simulations': n_simulations,
        },
        'matchups': matchups_out,
    }


def build_match_outcomes_payload(match_outcomes_df, n_simulations):
    """
    Wrap the match outcomes DataFrame into the JSON payload shape
    expected by the dashboard.

    Args:
        match_outcomes_df (pd.DataFrame): Output of build_match_outcomes().
        n_simulations (int): Number of simulation runs.

    Returns:
        dict: Ready to pass to export_json().
    """
    return {
        'meta': {
            'last_updated':  str(date.today()),
            'n_simulations': n_simulations,
        },
        'matches': match_outcomes_df.to_dict(orient='records'),
    }


def build_conditional_payload(conditional_df, n_simulations):
    """
    Wrap the conditional probabilities DataFrame into the JSON payload
    shape expected by the dashboard.

    Reshapes the flat DataFrame into a nested structure:
        team → list of {matchday_3_result, n_sims, p_1st, ..., p_advances}

    Args:
        conditional_df (pd.DataFrame): Output of build_conditional_probabilities().
        n_simulations (int): Number of simulation runs.

    Returns:
        dict: Ready to pass to export_json().
    """
    if conditional_df.empty:
        return {
            'meta': {
                'last_updated':     str(date.today()),
                'n_simulations':    n_simulations,
                'show_conditional': False,
            },
            'teams': [],
        }

    teams_out = []
    for (team_id, team_name, group), group_df in conditional_df.groupby(
        ['team_id', 'team_name', 'group']
    ):
        outcomes = group_df[[
            'matchday_3_result', 'n_sims',
            'p_1st', 'p_2nd', 'p_3rd', 'p_4th', 'p_advances',
        ]].to_dict(orient='records')

        # Sort outcomes: win → draw → loss for consistent display
        order = {'win': 0, 'draw': 1, 'loss': 2}
        outcomes.sort(key=lambda x: order.get(x['matchday_3_result'], 9))

        teams_out.append({
            'team_id':   int(team_id),
            'team_name': team_name,
            'group':     group,
            'outcomes':  outcomes,
        })

    teams_out.sort(key=lambda t: (t['group'], t['team_name']))

    return {
        'meta': {
            'last_updated':     str(date.today()),
            'n_simulations':    n_simulations,
            'show_conditional': SHOW_CONDITIONAL,
        },
        'teams': teams_out,
    }


# ============================================================
# SECTION 2 — EXPORT UTILITY
# ============================================================

def export_json(payload, filename):
    """
    Write a dict to a JSON file inside OUTPUT_DIR.
    Creates the directory if it does not exist.

    Args:
        payload (dict): Data to serialise.
        filename (str): Filename only, e.g. 'probabilities.json'.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)

    with open(path, 'w') as f:
        json.dump(payload, f, indent=2)

    size_kb = os.path.getsize(path) / 1024
    logger.info("Exported %-40s (%.1f KB)", path, size_kb)


# ============================================================
# SECTION 3 — MAIN
# ============================================================

if __name__ == '__main__':

    # ── Load data ────────────────────────────────────────────
    logger.info("Loading data from database…")
    matches_df, teams_df = wc.load_data()
    logger.info("Loaded %d matches, %d teams", len(matches_df), len(teams_df))

    # ── Run simulation ───────────────────────────────────────
    logger.info("Starting Monte Carlo — %d simulations", N_SIMULATIONS)
    (
        placement_counts,
        r32_counts,
        all_goals,
        match_results,
        match_results_conditional,
    ) = run_monte_carlo(matches_df, teams_df, N_SIMULATIONS)

    # ── Build DataFrames ─────────────────────────────────────
    logger.info("Building output DataFrames…")

    team_probs_df   = build_team_probabilities(placement_counts, teams_df, N_SIMULATIONS)
    r32_probs_df    = build_r32_opponent_probabilities(r32_counts, teams_df, N_SIMULATIONS)
    match_outcomes_df = build_match_outcomes(match_results, teams_df)
    conditional_df  = build_conditional_probabilities(match_results_conditional)

    # Diagnostics — pass completed matches for real vs simulated comparison
    completed_matches = matches_df[
        (matches_df['stage_id'] == 1) &
        (matches_df['status'] == 'completed')
    ]
    diagnostics = build_goals_diagnostics(
        all_goals,
        completed_matches_df=completed_matches if len(completed_matches) > 0 else None,
        mean_goals=MEAN_GOALS,
    )
    logger.info("Goals diagnostics summary: %s", diagnostics['summary'])
    logger.info("Goals diagnostics distribution of sim goals: %s", diagnostics['simulated_dist'])

    # ── Build JSON payloads ──────────────────────────────────
    logger.info("Building JSON payloads…")

    prob_payload        = build_probabilities_payload(team_probs_df, matches_df, N_SIMULATIONS)
    r32_payload         = build_r32_payload(r32_probs_df, N_SIMULATIONS)
    outcomes_payload    = build_match_outcomes_payload(match_outcomes_df, N_SIMULATIONS)
    conditional_payload = build_conditional_payload(conditional_df, N_SIMULATIONS)

    # ── Export ───────────────────────────────────────────────
    logger.info("Exporting JSON files to %s…", OUTPUT_DIR)

    export_json(prob_payload,        'probabilities.json')
    export_json(r32_payload,         'r32_opponents.json')
    export_json(outcomes_payload,    'match_outcomes.json')
    export_json(conditional_payload, 'conditional_outcomes.json')

    logger.info("Done.")