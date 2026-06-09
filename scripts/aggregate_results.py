"""
aggregate_results.py

Runs the simulator N times and aggregates results into probability tables.

Output DataFrames:
  - team_probabilities: per-team advancement and placement probabilities
  - r32_opponent_probabilities: per-team R32 opponent likelihood
  - goals_diagnostics: goals scored distribution for validation
"""

import pandas as pd
import numpy as np
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


def run_monte_carlo(matches_df, teams_df, n_simulations, scenarios):
    """
    Run the full simulation pipeline N times and accumulate raw counts.

    Args:
        matches_df: Full matches DataFrame from DB (all 104 rows)
        teams_df: Teams DataFrame from DB
        n_simulations: Number of simulation runs (e.g. 10000)
        scenarios: THIRD_PLACE_SCENARIOS_BY_GROUPS dict

    Returns:
        tuple: (placement_counts, r32_counts, all_goals)
            placement_counts: dict {team_id: {1: n, 2: n, 3: n, 4: n}}
            r32_counts: dict {team_id: {opponent_id: n}}
            all_goals: list of all simulated goal values (for diagnostics)
    """
    import wc_simulator as wc

    # Initialize counters
    team_ids = teams_df['id'].tolist()

    placement_counts = {
        tid: {'finish_1st': 0, 'finish_2nd': 0, 'finish_3rd': 0, 'finish_4th': 0,
              'advances': 0, 'eliminated': 0}
        for tid in team_ids
    }
    r32_counts = defaultdict(lambda: defaultdict(int))
    all_goals = []

    for i in range(n_simulations):
        if i % 1000 == 0:
            logger.info("Simulation run %d / %d", i, n_simulations)

        try:
            final_standings, third_place_df, scenario, r32_matchups, simulated_matches = wc.run_simulation(
                teams_df, matches_df
            )

            # Collect simulated goals
            all_goals.extend(simulated_matches['home_score'].tolist())
            all_goals.extend(simulated_matches['away_score'].tolist())
            
        except Exception as e:
            logger.warning("Simulation %d failed: %s", i, e)
            continue

        # --- Placement counts ---
        for _, row in final_standings.iterrows():
            tid = row['team_id']
            rank = int(row['overall_rank'])
            col = {1: 'finish_1st', 2: 'finish_2nd',
                   3: 'finish_3rd', 4: 'finish_4th'}.get(rank)
            if col:
                placement_counts[tid][col] += 1

        # --- Advancement counts ---
        # 1st and 2nd always advance
        auto_advance = final_standings[final_standings['overall_rank'].isin([1, 2])]['team_id'].tolist()
        for tid in auto_advance:
            placement_counts[tid]['advances'] += 1

        # Best 8 third-place teams advance
        if third_place_df is not None:
            advancing_third = third_place_df[third_place_df['advances']]['team_id'].tolist()
            for tid in advancing_third:
                placement_counts[tid]['advances'] += 1

            # Eliminated third-place teams
            eliminated_third = third_place_df[~third_place_df['advances']]['team_id'].tolist()
            for tid in eliminated_third:
                placement_counts[tid]['eliminated'] += 1

        # 4th place always eliminated
        fourth_place = final_standings[final_standings['overall_rank'] == 4]['team_id'].tolist()
        for tid in fourth_place:
            placement_counts[tid]['eliminated'] += 1

        # --- R32 opponent counts ---
        if r32_matchups:
            for matchup in r32_matchups:
                home_id = matchup['home_team_id']
                away_id = matchup['away_team_id']
                r32_counts[home_id][away_id] += 1
                r32_counts[away_id][home_id] += 1

        # --- Collect all simulated goals for diagnostics ---
        simulated = matches_df[
            (matches_df['stage_id'] == 1) &
            (matches_df['status'] == 'scheduled')
        ]
        # Note: you'll need run_simulation to return simulated_matches
        # for this to work — see diagnostics note below

    return placement_counts, r32_counts, all_goals


def build_team_probabilities(placement_counts, teams_df, n_simulations):
    """
    Convert raw placement counts into a probability DataFrame.

    Returns:
        pd.DataFrame with columns:
            team_id, team_name, group,
            p_1st, p_2nd, p_3rd, p_4th,
            p_advances, p_eliminated
    """
    rows = []
    team_lookup = teams_df.set_index('id')[['team_name', 'group_letter']].to_dict('index')

    for tid, counts in placement_counts.items():
        info = team_lookup.get(tid, {})
        rows.append({
            'team_id':      tid,
            'team_name':    info.get('team_name', 'Unknown'),
            'group':        info.get('group_letter', '?'),
            'p_1st':        round(counts['finish_1st'] / n_simulations, 4),
            'p_2nd':        round(counts['finish_2nd'] / n_simulations, 4),
            'p_3rd':        round(counts['finish_3rd'] / n_simulations, 4),
            'p_4th':        round(counts['finish_4th'] / n_simulations, 4),
            'p_advances':   round(counts['advances']   / n_simulations, 4),
            'p_eliminated': round(counts['eliminated'] / n_simulations, 4),
        })

    df = pd.DataFrame(rows).sort_values(['group', 'p_advances'], ascending=[True, False])
    return df.reset_index(drop=True)


def build_r32_opponent_probabilities(r32_counts, teams_df, n_simulations):
    """
    Convert raw R32 opponent counts into a probability DataFrame.

    Returns:
        r32_probs (pd.DataFrame):
            team_id, team_name, opponent_id, opponent_name, probability
        Sorted by team_name then probability descending.
        Only includes opponents with probability > 0.
    """
    team_lookup = teams_df.set_index('id')['team_name'].to_dict()
    rows = []

    for tid, opponent_counts in r32_counts.items():
        team_name = team_lookup.get(tid, 'Unknown')
        total = sum(opponent_counts.values())

        for opp_id, count in opponent_counts.items():
            rows.append({
                'team_id':       tid,
                'team_name':     team_name,
                'opponent_id':   opp_id,
                'opponent_name': team_lookup.get(opp_id, 'Unknown'),
                'probability':   round(count / n_simulations, 4),
            })

    df = pd.DataFrame(rows)
    df = df[df['probability'] > 0]
    return df.sort_values(['team_name', 'probability'], ascending=[True, False]).reset_index(drop=True)


def build_goals_diagnostics(all_simulated_goals, completed_matches_df=None, mean_goals=1.38):
    """
    Build a diagnostics report comparing simulated goal distribution
    to the theoretical Poisson distribution and (optionally) real results.

    Args:
        all_simulated_goals: flat list of all simulated goal values
                             collected across all simulation runs
        completed_matches_df: optional DataFrame of real completed matches
                              with home_score and away_score columns
        mean_goals: the Poisson mean used in the simulation

    Returns:
        dict with keys:
            'simulated_dist': DataFrame — goals 0-8, simulated frequency vs Poisson expected
            'real_dist': DataFrame — same for real matches (None if not provided)
            'summary': dict — mean, std, median for simulated and real
    """
    from scipy.stats import poisson as poisson_dist

    # --- Simulated distribution ---
    max_goals = 8
    sim_series = pd.Series(all_simulated_goals)
    sim_counts = sim_series.value_counts().sort_index()
    sim_total = len(sim_series)

    theoretical_probs = [poisson_dist.pmf(g, mean_goals) for g in range(max_goals + 1)]

    sim_dist = pd.DataFrame({
        'goals':              range(max_goals + 1),
        'simulated_freq':     [sim_counts.get(g, 0) / sim_total for g in range(max_goals + 1)],
        'poisson_expected':   theoretical_probs,
    })
    sim_dist['simulated_freq'] = sim_dist['simulated_freq'].round(4)
    sim_dist['poisson_expected'] = sim_dist['poisson_expected'].round(4)
    sim_dist['difference'] = (sim_dist['simulated_freq'] - sim_dist['poisson_expected']).round(4)

    summary = {
        'simulated_mean':   round(float(sim_series.mean()), 4),
        'simulated_std':    round(float(sim_series.std()), 4),
        'simulated_median': round(float(sim_series.median()), 4),
        'poisson_mean':     mean_goals,
        'n_samples':        sim_total,
    }

    # --- Real match distribution (once games are played) ---
    real_dist = None
    if completed_matches_df is not None and len(completed_matches_df) > 0:
        real_goals = pd.concat([
            completed_matches_df['home_score'],
            completed_matches_df['away_score']
        ]).dropna().astype(int)

        real_total = len(real_goals)
        real_counts = real_goals.value_counts().sort_index()

        real_dist = pd.DataFrame({
            'goals':          range(max_goals + 1),
            'real_freq':      [real_counts.get(g, 0) / real_total for g in range(max_goals + 1)],
            'poisson_expected': theoretical_probs,
        })
        real_dist['real_freq'] = real_dist['real_freq'].round(4)
        real_dist['difference'] = (real_dist['real_freq'] - real_dist['poisson_expected']).round(4)

        summary.update({
            'real_mean':   round(float(real_goals.mean()), 4),
            'real_std':    round(float(real_goals.std()), 4),
            'real_median': round(float(real_goals.median()), 4),
            'n_real_goals': real_total,
        })

    return {
        'simulated_dist': sim_dist,
        'real_dist':      real_dist,
        'summary':        summary,
    }