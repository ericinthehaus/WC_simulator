'''Update_Scores.py'''
import os
import sqlite3
import requests
import pandas as pd
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

DB_PATH = 'worldcup2026.db'
API_KEY = os.getenv('FOOTBALL_API_KEY')       # store in env var, not hardcoded
API_MATCH_URL = 'https://api.football-data.org/v4/competitions/WC/matches'

API_STANDINGS_URL = "http://api.football-data.org/v4/competitions/WC/teams"

def fetch_live_teams(api_key=API_KEY):
    headers = {'X-Auth-Token': api_key}
    response = requests.get(API_STANDINGS_URL, headers=headers)

    if response.status_code != 200:
        logger.error("API request failed: %s %s", response.status_code, response.text)
        return []

    data = response.json()
    return data.get('teams', [])

def fetch_live_scores(api_key=API_KEY):
    """
    Fetch all World Cup match results from football-data.org.

    Returns:
        list[dict]: Raw match data from the API, one dict per match.
        Returns empty list if the request fails.
    """
    headers = {'X-Auth-Token': api_key}
    response = requests.get(API_MATCH_URL, headers=headers)

    if response.status_code != 200:
        logger.error("API request failed: %s %s", response.status_code, response.text)
        return []

    data = response.json()
    return data.get('matches', [])


def parse_match_result(api_match):
    """
    Extract the fields we need from a single API match object.

    Args:
        api_match (dict): A single match dict from the API response.

    Returns:
        dict with keys: api_match_id, home_score, away_score, result, status
        Returns None if the match is not yet finished.
    """
    status = api_match.get('status')

    # Only process matches that are finished
    if status != 'FINISHED':
        return None

    score = api_match.get('score', {}).get('fullTime', {})
    home_score = score.get('home')
    away_score = score.get('away')

    if home_score is None or away_score is None:
        logger.warning("Match %s has FINISHED status but missing scores", api_match.get('id'))
        return None

    if home_score > away_score:
        result = 'home_win'
    elif away_score > home_score:
        result = 'away_win'
    else:
        result = 'draw'

    return {
        'api_match_id': api_match['id'],
        'home_score': int(home_score),
        'away_score': int(away_score),
        'result': result,
        'status': 'completed'
    }


def update_completed_matches(db_path=DB_PATH, api_key=API_KEY):
    """
    Main update function. Fetches live scores and writes completed results to DB.

    Only updates matches that are currently 'scheduled' in the DB —
    never overwrites a row already marked 'completed'.

    Args:
        db_path (str): Path to the SQLite database.
        api_key (str): football-data.org API key.

    Returns:
        int: Number of matches updated.
    """
    raw_matches = fetch_live_scores(api_key)
    if not raw_matches:
        logger.info("No matches returned from API.")
        return 0

    conn = sqlite3.connect(db_path)

    updated = 0
    for api_match in raw_matches:
        parsed = parse_match_result(api_match)
        if parsed is None:
            continue  # not finished yet

        api_id = parsed['api_match_id']
        home_score = parsed['home_score']
        away_score = parsed['away_score']
        result     = parsed['result']
        
        cursor = conn.execute("""
                     UPDATE matches
                     SET home_score = ?, away_score = ?, result = ?, status = 'completed'
                     WHERE id = ? AND status != 'completed'
                     """, (home_score, away_score, result, api_id))

        if cursor.rowcount > 0:
            updated += 1

    conn.commit()
    conn.close()

    logger.info("Updated %d matches in the database.", updated)
    return updated


if __name__ == "__main__":
    import os
    logging.basicConfig(level=logging.INFO)
    # api_key = os.getenv("FOOTBALL_API_KEY")  # reads from environment variable
    update_completed_matches(api_key=API_KEY)