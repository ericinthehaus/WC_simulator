"""
constants.py

Shared simulation constants used across wc_simulator.py
and run_analysis_export_results.py.

Adjust MEAN_GOALS here to change the scoring model globally.
"""

MEAN_GOALS    = 1.38   # Poisson mean for goals per team per match
N_SIMULATIONS = 10 # Number of Monte Carlo runs
DB_PATH       = 'worldcup2026.db'
OUTPUT_DIR    = 'docs/output'