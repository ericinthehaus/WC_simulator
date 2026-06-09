"""
run_analysis_export_results.py

Runs the World Cup 2026 Monte Carlo simulation N times, aggregates results,
and exports two JSON files for the dashboard:

    docs/output/probabilities.json   — group stage placement + advancement odds
    docs/output/r32_opponents.json   — likely Round of 32 opponents per team

Usage:
    python run_analysis_export_results.py

Outputs:
    docs/output/probabilities.json
    docs/output/r32_opponents.json
"""

import json
import logging
import os
from collections import defaultdict
from datetime import date

import wc_simulator as wc

# ============================================================
# CONFIG
# ============================================================

N_SIMULATIONS = 10_000 
OUTPUT_DIR    = 'docs/output'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(message)s',
    datefmt='%H:%M:%S'
)
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
            placement_counts (dict): {team_id: {stat: count}} raw finish tallies.
            r32_counts (defaultdict): {team_id: {opponent_id: count}} R32 matchup tallies.
            all_goals (list[int]): Every simulated goal value — for diagnostics.
    """
    team_ids = teams_df['id'].tolist()

    # Initialise placement counters for all 48 teams
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

    # Nested defaultdict — convert before JSON serialisation (see export section)
    r32_counts = defaultdict(lambda: defaultdict(int))
    all_goals  = []

    failed = 0

    for i in range(n_simulations):
        if i > 0 and i % 1_000 == 0:
            logger.info("Simulation %d / %d  (failed so far: %d)", i, n_simulations, failed)

        try:
            final_standings, third_place_df, scenario, r32_matchups, simulated_matches = \
                wc.run_simulation(teams_df, matches_df)
        except Exception as e:
            logger.warning("Simulation %d failed: %s", i, e)
            failed += 1
            continue

        # ── Placement counts ────────────────────────────────
        rank_to_col = {1: 'finish_1st', 2: 'finish_2nd',
                       3: 'finish_3rd', 4: 'finish_4th'}

        for _, row in final_standings.iterrows():
            tid  = row['team_id']
            rank = int(row['overall_rank'])

            if tid not in placement_counts:
                continue

            # Placement
            col = rank_to_col.get(rank)
            if col:
                placement_counts[tid][col] += 1

            # Advancement — 1st and 2nd auto-advance
            if rank in (1, 2):
                placement_counts[tid]['advances'] += 1

            # 4th place always eliminated
            if rank == 4:
                placement_counts[tid]['eliminated'] += 1

        # Best 8 third-place teams advance; bottom 4 eliminated
        if third_place_df is not None:
            for _, row in third_place_df.iterrows():
                tid = row['team_id']
                if tid not in placement_counts:
                    continue
                if row['advances']:
                    placement_counts[tid]['advances'] += 1
                else:
                    placement_counts[tid]['eliminated'] += 1

        # ── R32 opponent counts ─────────────────────────────
        if r32_matchups:
            for matchup in r32_matchups:
                home_id = matchup['home_team_id']
                away_id = matchup['away_team_id']
                r32_counts[home_id][away_id] += 1
                r32_counts[away_id][home_id] += 1

        # ── Collect simulated goals for diagnostics ─────────
        if simulated_matches is not None:
            all_goals.extend(simulated_matches['home_score'].dropna().astype(int).tolist())
            all_goals.extend(simulated_matches['away_score'].dropna().astype(int).tolist())

    logger.info(
        "Monte Carlo complete — %d runs succeeded, %d failed",
        n_simulations - failed, failed
    )

    return placement_counts, r32_counts, all_goals


# ============================================================
# SECTION 2 — BUILD OUTPUT STRUCTURES
# ============================================================

def build_probabilities_payload(placement_counts, teams_df, matches_df, n_simulations):
    """
    Build the full payload for probabilities.json.

    Structure:
        {
            "meta": { last_updated, n_simulations, matches_completed, matches_total },
            "teams": [ { team_id, team_name, group, p_1st, p_2nd, p_3rd, p_4th,
                         p_advances, p_eliminated }, ... ]
        }

    Args:
        placement_counts (dict): Raw counts from run_monte_carlo().
        teams_df (pd.DataFrame): Teams table.
        matches_df (pd.DataFrame): Full matches table — used to compute meta counts.
        n_simulations (int): Number of simulation runs (divisor for probabilities).

    Returns:
        dict: Ready to serialise with json.dump().
    """
    team_lookup = teams_df.set_index('id')[['team_name', 'group_letter', 'crest_url']].to_dict('index')

    # Meta — live counts from the DB
    group_matches    = matches_df[matches_df['stage_id'] == 1]
    matches_completed = int((group_matches['status'] == 'completed').sum())
    matches_total     = int(len(group_matches))

    teams_out = []
    for tid, counts in placement_counts.items():
        info = team_lookup.get(tid, {})
        teams_out.append({
            'team_id':      int(tid),
            'crest_url':    info.get('crest_url',    None),   # ← add this line
            'team_name':    info.get('team_name',    'Unknown'),
            'group':        info.get('group_letter', '?'),
            'p_1st':        round(counts['finish_1st']  / n_simulations, 4),
            'p_2nd':        round(counts['finish_2nd']  / n_simulations, 4),
            'p_3rd':        round(counts['finish_3rd']  / n_simulations, 4),
            'p_4th':        round(counts['finish_4th']  / n_simulations, 4),
            'p_advances':   round(counts['advances']    / n_simulations, 4),
            'p_eliminated': round(counts['eliminated']  / n_simulations, 4),
        })

    # Sort by group letter then descending advancement probability
    teams_out.sort(key=lambda t: (t['group'], -t['p_advances']))

    return {
        'meta': {
            'last_updated':      str(date.today()),
            'n_simulations':     n_simulations,
            'matches_completed': matches_completed,
            'matches_total':     matches_total,
        },
        'teams': teams_out,
    }


def build_r32_payload(r32_counts, teams_df, n_simulations):
    """
    Build the full payload for r32_opponents.json.

    Structure:
        {
            "meta": { last_updated, n_simulations },
            "matchups": [
                {
                    "team_id": 762,
                    "team_name": "Argentina",
                    "group": "J",
                    "opponents": [
                        { "opponent_id": 8049, "opponent_name": "Jordan",
                          "opponent_group": "J", "probability": 0.312 },
                        ...
                    ]
                },
                ...
            ]
        }

    Args:
        r32_counts (defaultdict): Raw counts from run_monte_carlo().
        teams_df (pd.DataFrame): Teams table.
        n_simulations (int): Number of simulation runs.

    Returns:
        dict: Ready to serialise with json.dump().
    """
    team_lookup = teams_df.set_index('id')[['team_name', 'group_letter']].to_dict('index')

    matchups_out = []

    for tid, opp_counts in r32_counts.items():
        info = team_lookup.get(tid, {})

        opponents = []
        for opp_id, count in opp_counts.items():
            prob = round(count / n_simulations, 4)
            if prob == 0:
                continue
            opp_info = team_lookup.get(opp_id, {})
            opponents.append({
                'opponent_id':    int(opp_id),
                'opponent_name':  opp_info.get('team_name',    'Unknown'),
                'opponent_group': opp_info.get('group_letter', '?'),
                'probability':    prob,
            })

        # Sort opponents by probability descending
        opponents.sort(key=lambda x: -x['probability'])

        matchups_out.append({
            'team_id':   int(tid),
            'team_name': info.get('team_name',    'Unknown'),
            'group':     info.get('group_letter', '?'),
            'opponents': opponents,
        })

    # Sort teams alphabetically by name
    matchups_out.sort(key=lambda t: t['team_name'])

    return {
        'meta': {
            'last_updated':  str(date.today()),
            'n_simulations': n_simulations,
        },
        'matchups': matchups_out,
    }


# ============================================================
# SECTION 3 — EXPORT
# ============================================================

def export_json(payload, filename):
    """
    Write a dict to a JSON file inside OUTPUT_DIR.

    Creates the output directory if it doesn't exist.

    Args:
        payload (dict): Data to serialise.
        filename (str): Filename only, e.g. 'probabilities.json'.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)

    with open(path, 'w') as f:
        json.dump(payload, f, indent=2)

    size_kb = os.path.getsize(path) / 1024
    logger.info("Exported %s  (%.1f KB)", path, size_kb)


# ============================================================
# SECTION 4 — MAIN
# ============================================================

if __name__ == '__main__':

    # ── Load data ────────────────────────────────────────────
    logger.info("Loading data from database…")
    matches_df, teams_df = wc.load_data()
    logger.info(
        "Loaded %d matches, %d teams",
        len(matches_df), len(teams_df)
    )

    # ── Run simulation ───────────────────────────────────────
    logger.info("Starting Monte Carlo — %d simulations", N_SIMULATIONS)
    placement_counts, r32_counts, all_goals = run_monte_carlo(
        matches_df, teams_df, N_SIMULATIONS
    )

    # ── Build payloads ───────────────────────────────────────
    logger.info("Building output payloads…")
    prob_payload = build_probabilities_payload(
        placement_counts, teams_df, matches_df, N_SIMULATIONS
    )
    r32_payload = build_r32_payload(
        r32_counts, teams_df, N_SIMULATIONS
    )

    # ── Export ───────────────────────────────────────────────
    export_json(prob_payload, 'probabilities.json')
    export_json(r32_payload,  'r32_opponents.json')

    logger.info("Done.")