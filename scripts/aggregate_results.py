"""
aggregate_results.py

Library of aggregation functions for the WC 2026 Monte Carlo simulation.
All functions in this file are imported and called by run_analysis_export_results.py.

Functions:
    run_monte_carlo                  — runs N simulations, accumulates raw counts
    collect_conditional_outcomes     — per-simulation matchday-3 conditional rows
    build_team_probabilities         — placement + advancement probabilities per team
    build_r32_opponent_probabilities — R32 likely opponents per team
    build_match_outcomes             — W/D/L probabilities and avg scores per match
    build_conditional_probabilities  — placement probabilities conditioned on matchday-3 result
    build_goals_diagnostics          — simulated vs theoretical goal distribution
"""

import pandas as pd
import numpy as np
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


# ============================================================
# SECTION 1 — MONTE CARLO RUNNER
# ============================================================

def run_monte_carlo(matches_df, teams_df, n_simulations):
    """
    Run the full simulation pipeline N times and accumulate raw counts.

    Args:
        matches_df (pd.DataFrame): Full matches table from DB (all 104 rows).
        teams_df (pd.DataFrame): Teams table from DB.
        n_simulations (int): Number of simulation runs.

    Returns:
        tuple:
            placement_counts (dict): {team_id: {stat: count}}
            r32_counts (defaultdict): {team_id: {opponent_id: count}}
            all_goals (list[int]): Every simulated goal value.
            match_results (list[pd.DataFrame]): One DataFrame per run
                containing simulated match rows.
            match_results_conditional (list[dict]): Per-team matchday-3
                result rows for conditional probability computation.
    """
    import wc_simulator as wc

    team_ids = teams_df['id'].tolist()

    # ── Initialise counters ──────────────────────────────────
    placement_counts = {
        tid: {
            'finish_1st':  0,
            'finish_2nd':  0,
            'finish_3rd':  0,
            'finish_4th':  0,
            'advances':    0,
            'eliminated':  0,
        }
        for tid in team_ids
    }
    r32_counts               = defaultdict(lambda: defaultdict(int))
    all_goals                = []
    match_results            = []
    match_results_conditional = []
    failed                   = 0

    rank_to_col = {
        1: 'finish_1st',
        2: 'finish_2nd',
        3: 'finish_3rd',
        4: 'finish_4th',
    }

    for i in range(n_simulations):
        if i > 0 and i % 1_000 == 0:
            logger.info(
                "Simulation %d / %d  (failed so far: %d)",
                i, n_simulations, failed
            )

        try:
            final_standings, third_place_df, scenario, r32_matchups, simulated_matches = \
                wc.run_simulation(teams_df, matches_df)
        except Exception as e:
            logger.warning("Simulation %d failed: %s", i, e)
            failed += 1
            continue

        # ── Goals for diagnostics ────────────────────────────
        all_goals.extend(simulated_matches['home_score'].dropna().astype(int).tolist())
        all_goals.extend(simulated_matches['away_score'].dropna().astype(int).tolist())

        # ── Placement + advancement counts ───────────────────
        for _, row in final_standings.iterrows():
            tid  = row['team_id']
            rank = int(row['overall_rank'])

            if tid not in placement_counts:
                continue

            col = rank_to_col.get(rank)
            if col:
                placement_counts[tid][col] += 1

            # 1st and 2nd auto-advance
            if rank in (1, 2):
                placement_counts[tid]['advances'] += 1

            # 4th always eliminated
            if rank == 4:
                placement_counts[tid]['eliminated'] += 1

        # 3rd-place teams — advance or eliminate based on cross-group ranking
        if third_place_df is not None:
            for _, row in third_place_df.iterrows():
                tid = row['team_id']
                if tid not in placement_counts:
                    continue
                if row['advances']:
                    placement_counts[tid]['advances'] += 1
                else:
                    placement_counts[tid]['eliminated'] += 1

        # ── R32 opponent counts ──────────────────────────────
        if r32_matchups:
            for matchup in r32_matchups:
                home_id = matchup['home_team_id']
                away_id = matchup['away_team_id']
                r32_counts[home_id][away_id] += 1
                r32_counts[away_id][home_id] += 1

        # ── Match results accumulator ────────────────────────
        sim_copy = simulated_matches.copy()
        sim_copy['simulation_id'] = i
        match_results.append(sim_copy)

        # ── Conditional outcomes (matchday 3) ────────────────
        collect_conditional_outcomes(
            simulation_id=i,
            simulated_matches=simulated_matches,
            final_standings=final_standings,
            third_place_df=third_place_df,
            match_results_conditional=match_results_conditional,
        )

    logger.info(
        "Monte Carlo complete — %d runs succeeded, %d failed",
        n_simulations - failed, failed
    )

    return (
        placement_counts,
        r32_counts,
        all_goals,
        match_results,
        match_results_conditional,
    )


# ============================================================
# SECTION 2 — CONDITIONAL OUTCOMES COLLECTOR
# ============================================================

def collect_conditional_outcomes(
    simulation_id,
    simulated_matches,
    final_standings,
    third_place_df,
    match_results_conditional,
):
    """
    For one simulation run, record each team's matchday-3 result alongside
    their final placement. Appends rows to match_results_conditional in place.

    Only processes matchday-3 matches that were simulated (status == 'simulated').
    Real completed matches are excluded — their outcomes are already known.

    Args:
        simulation_id (int): Current simulation run index.
        simulated_matches (pd.DataFrame): Output of simulate_group_stage_matches*().
            Must contain: matchday, home_team_id, away_team_id, result, status.
        final_standings (pd.DataFrame): Output of apply_all_tiebreakers().
            Must contain: team_id, team_name, group, overall_rank.
        third_place_df (pd.DataFrame | None): Ranked third-place teams with
            advances column. Used to resolve advancement for rank-3 teams.
        match_results_conditional (list): Accumulator — rows appended in place.

    Returns:
        None.
    """
    # ── Build standings lookup {team_id: {...}} ──────────────
    standings_lookup = {}
    for _, row in final_standings.iterrows():
        standings_lookup[row['team_id']] = {
            'team_name':  row['team_name'],
            'group':      row['group'],
            'final_rank': int(row['overall_rank']),
            'advances':   int(row['overall_rank']) <= 2,
        }

    # ── Refine advances for 3rd-place teams ─────────────────
    if third_place_df is not None:
        advancing_thirds = set(
            third_place_df[third_place_df['advances']]['team_id'].tolist()
        )
        for tid, info in standings_lookup.items():
            if info['final_rank'] == 3:
                info['advances'] = tid in advancing_thirds

    # ── Filter to simulated matchday-3 matches only ──────────
    md3 = simulated_matches[
        (simulated_matches['matchday'] == 3) &
        (simulated_matches['status'] == 'simulated')
    ]

    for _, match in md3.iterrows():
        home_id    = match['home_team_id']
        away_id    = match['away_team_id']
        raw_result = match['result']

        # Translate to per-team perspective
        if raw_result == 'home_win':
            home_result, away_result = 'win', 'loss'
        elif raw_result == 'away_win':
            home_result, away_result = 'loss', 'win'
        else:
            home_result, away_result = 'draw', 'draw'

        for team_id, result in [(home_id, home_result), (away_id, away_result)]:
            info = standings_lookup.get(team_id)
            if info is None:
                continue

            match_results_conditional.append({
                'simulation_id':     simulation_id,
                'team_id':           team_id,
                'team_name':         info['team_name'],
                'group':             info['group'],
                'matchday_3_result': result,
                'final_rank':        info['final_rank'],
                'advances':          info['advances'],
            })


# ============================================================
# SECTION 3 — BUILD OUTPUT DATAFRAMES
# ============================================================

def build_team_probabilities(placement_counts, teams_df, n_simulations):
    """
    Convert raw placement counts into a probability DataFrame.

    Args:
        placement_counts (dict): Raw counts from run_monte_carlo().
        teams_df (pd.DataFrame): Teams table — provides team_name,
            group_letter, crest_url per team_id.
        n_simulations (int): Divisor for probability calculation.

    Returns:
        pd.DataFrame with columns:
            team_id, team_name, group, crest_url,
            p_1st, p_2nd, p_3rd, p_4th,
            p_advances, p_eliminated
        Sorted by group then p_advances descending.
    """
    team_lookup = teams_df.set_index('id')[
        ['team_name', 'group_letter', 'crest_url']
    ].to_dict('index')

    rows = []
    for tid, counts in placement_counts.items():
        info = team_lookup.get(tid, {})
        rows.append({
            'team_id':      int(tid),
            'team_name':    info.get('team_name',    'Unknown'),
            'group':        info.get('group_letter', '?'),
            'crest_url':    info.get('crest_url',    None),
            'p_1st':        round(counts['finish_1st']  / n_simulations, 4),
            'p_2nd':        round(counts['finish_2nd']  / n_simulations, 4),
            'p_3rd':        round(counts['finish_3rd']  / n_simulations, 4),
            'p_4th':        round(counts['finish_4th']  / n_simulations, 4),
            'p_advances':   round(counts['advances']    / n_simulations, 4),
            'p_eliminated': round(counts['eliminated']  / n_simulations, 4),
        })

    df = pd.DataFrame(rows).sort_values(
        ['group', 'p_advances'], ascending=[True, False]
    )
    return df.reset_index(drop=True)


def build_r32_opponent_probabilities(r32_counts, teams_df, n_simulations):
    """
    Convert raw R32 opponent counts into a probability DataFrame.

    Args:
        r32_counts (defaultdict): Raw counts from run_monte_carlo().
        teams_df (pd.DataFrame): Teams table.
        n_simulations (int): Divisor for probability calculation.

    Returns:
        pd.DataFrame with columns:
            team_id, team_name, group, crest_url,
            opponent_id, opponent_name, opponent_group, probability
        Sorted by team_name then probability descending.
        Opponents with probability < 0.001 are excluded.
    """
    team_lookup = teams_df.set_index('id')[
        ['team_name', 'group_letter', 'crest_url']
    ].to_dict('index')

    rows = []
    for tid, opponent_counts in r32_counts.items():
        info = team_lookup.get(tid, {})

        for opp_id, count in opponent_counts.items():
            prob = round(count / n_simulations, 4)
            if prob < 0.001:
                continue

            opp_info = team_lookup.get(opp_id, {})
            rows.append({
                'team_id':        int(tid),
                'team_name':      info.get('team_name',    'Unknown'),
                'group':          info.get('group_letter', '?'),
                'crest_url':      info.get('crest_url',    None),
                'opponent_id':    int(opp_id),
                'opponent_name':  opp_info.get('team_name',    'Unknown'),
                'opponent_group': opp_info.get('group_letter', '?'),
                'probability':    prob,
            })

    df = pd.DataFrame(rows)
    return df.sort_values(
        ['team_name', 'probability'], ascending=[True, False]
    ).reset_index(drop=True)


def build_match_outcomes(match_results, teams_df):
    """
    Aggregate simulated match outcomes across all simulation runs.

    For each group stage match, computes W/D/L probabilities, average
    scores, and the most common exact scoreline.

    Args:
        match_results (list[pd.DataFrame]): One DataFrame per simulation run.
            Each must contain: id, home_team_id, away_team_id,
            home_score, away_score, result, group_letter (if available).
        teams_df (pd.DataFrame): Teams table for name lookups.

    Returns:
        pd.DataFrame with columns:
            match_id, home_team_name, away_team_name, group_letter,
            home_win_pct, draw_pct, away_win_pct,
            avg_home_score, avg_away_score, most_common_score
        Returns empty DataFrame if match_results is empty.
    """
    if not match_results:
        return pd.DataFrame()

    team_lookup = teams_df.set_index('id')['team_name'].to_dict()
    all_matches = pd.concat(match_results, ignore_index=True)

    rows = []
    for match_id, group_df in all_matches.groupby('id'):
        total = len(group_df)

        result_counts = group_df['result'].value_counts()
        home_id = group_df['home_team_id'].iloc[0]
        away_id = group_df['away_team_id'].iloc[0]

        scorelines = (
            group_df['home_score'].astype(str) + '-' +
            group_df['away_score'].astype(str)
        )

        rows.append({
            'match_id':          int(match_id),
            'home_team_name':    team_lookup.get(home_id, 'Unknown'),
            'away_team_name':    team_lookup.get(away_id, 'Unknown'),
            'group_letter':      group_df['group_letter'].iloc[0]
                                 if 'group_letter' in group_df.columns else '?',
            'home_win_pct':      round(result_counts.get('home_win', 0) / total, 4),
            'draw_pct':          round(result_counts.get('draw',     0) / total, 4),
            'away_win_pct':      round(result_counts.get('away_win', 0) / total, 4),
            'avg_home_score':    round(group_df['home_score'].mean(), 2),
            'avg_away_score':    round(group_df['away_score'].mean(), 2),
            'most_common_score': scorelines.value_counts().idxmax(),
        })

    return pd.DataFrame(rows).sort_values('match_id').reset_index(drop=True)


def build_conditional_probabilities(match_results_conditional):
    """
    Aggregate raw conditional rows into per-team, per-result probabilities.

    Answers: "Given Team X wins/draws/loses their final group match,
    what is the probability they finish 1st/2nd/3rd/4th and advance?"

    Args:
        match_results_conditional (list[dict]): Raw rows from
            collect_conditional_outcomes() across all simulation runs.

    Returns:
        pd.DataFrame with columns:
            team_id, team_name, group, matchday_3_result,
            n_sims, p_1st, p_2nd, p_3rd, p_4th, p_advances
        Sorted by group, team_name, matchday_3_result.
        Returns empty DataFrame if input is empty.
    """
    if not match_results_conditional:
        logger.info(
            "No conditional outcomes collected — matchday-3 matches "
            "may all be completed already or matchday column is missing."
        )
        return pd.DataFrame()

    df = pd.DataFrame(match_results_conditional)

    rows = []
    for (team_id, result), group_df in df.groupby(['team_id', 'matchday_3_result']):
        n = len(group_df)
        rows.append({
            'team_id':           int(team_id),
            'team_name':         group_df['team_name'].iloc[0],
            'group':             group_df['group'].iloc[0],
            'matchday_3_result': result,
            'n_sims':            n,
            'p_1st':             round((group_df['final_rank'] == 1).sum() / n, 4),
            'p_2nd':             round((group_df['final_rank'] == 2).sum() / n, 4),
            'p_3rd':             round((group_df['final_rank'] == 3).sum() / n, 4),
            'p_4th':             round((group_df['final_rank'] == 4).sum() / n, 4),
            'p_advances':        round(group_df['advances'].sum() / n, 4),
        })

    return pd.DataFrame(rows).sort_values(
        ['group', 'team_name', 'matchday_3_result']
    ).reset_index(drop=True)


# ============================================================
# SECTION 4 — GOALS DIAGNOSTICS
# ============================================================

def build_goals_diagnostics(all_simulated_goals, completed_matches_df=None, mean_goals=1.38):
    """
    Build a diagnostics report comparing simulated goal distribution
    to the theoretical Poisson and (optionally) real match results.

    Args:
        all_simulated_goals (list[int]): All simulated goal values
            from run_monte_carlo().
        completed_matches_df (pd.DataFrame | None): Real completed matches
            with home_score and away_score columns. Pass None before
            any matches are played.
        mean_goals (float): Poisson mean used in the simulation.

    Returns:
        dict with keys:
            'simulated_dist' (pd.DataFrame): goals 0-8, simulated
                frequency vs Poisson expected, difference.
            'real_dist' (pd.DataFrame | None): same for real matches.
            'summary' (dict): mean, std, median for simulated and real.
    """
    from scipy.stats import poisson as poisson_dist

    max_goals = 8
    sim_series = pd.Series(all_simulated_goals)
    sim_counts = sim_series.value_counts().sort_index()
    sim_total  = len(sim_series)

    theoretical = [poisson_dist.pmf(g, mean_goals) for g in range(max_goals + 1)]

    sim_dist = pd.DataFrame({
        'goals':            range(max_goals + 1),
        'simulated_freq':   [round(sim_counts.get(g, 0) / sim_total, 4)
                             for g in range(max_goals + 1)],
        'poisson_expected': [round(p, 4) for p in theoretical],
    })
    sim_dist['difference'] = (
        sim_dist['simulated_freq'] - sim_dist['poisson_expected']
    ).round(4)

    summary = {
        'simulated_mean':   round(float(sim_series.mean()),   4),
        'simulated_std':    round(float(sim_series.std()),    4),
        'simulated_median': round(float(sim_series.median()), 4),
        'poisson_mean':     mean_goals,
        'n_samples':        sim_total,
    }

    real_dist = None
    if completed_matches_df is not None and len(completed_matches_df) > 0:
        real_goals = pd.concat([
            completed_matches_df['home_score'],
            completed_matches_df['away_score'],
        ]).dropna().astype(int)

        real_total  = len(real_goals)
        real_counts = real_goals.value_counts().sort_index()

        real_dist = pd.DataFrame({
            'goals':            range(max_goals + 1),
            'real_freq':        [round(real_counts.get(g, 0) / real_total, 4)
                                 for g in range(max_goals + 1)],
            'poisson_expected': [round(p, 4) for p in theoretical],
        })
        real_dist['difference'] = (
            real_dist['real_freq'] - real_dist['poisson_expected']
        ).round(4)

        summary.update({
            'real_mean':    round(float(real_goals.mean()),   4),
            'real_std':     round(float(real_goals.std()),    4),
            'real_median':  round(float(real_goals.median()), 4),
            'n_real_goals': real_total,
        })

    return {
        'simulated_dist': sim_dist,
        'real_dist':      real_dist,
        'summary':        summary,
    }