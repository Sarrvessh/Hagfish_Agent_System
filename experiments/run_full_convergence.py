"""
Full Convergence Analysis - All 8 HPOBench Datasets

USAGE:
    python run_full_convergence.py

This will run convergence analysis on all 8 datasets:
- australian, blood_transfusion, car, credit_g
- segment, vehicle, kr_vs_kp, phoneme

Expected runtime: ~10-15 minutes for 10 seeds × 50 rounds × 8 datasets
"""

import subprocess
import sys

# Full benchmark: All 8 datasets, 10 seeds, 50 rounds
cmd = [
    sys.executable,
    "convergence_analysis.py",
    "--seeds", "10",
    "--rounds", "50",
    "--alpha", "0.3",
    "--datasets",
    "australian",
    "blood_transfusion",
    "car",
    "credit_g",
    "segment",
    "vehicle",
    "kr_vs_kp",
    "phoneme",
    "--output-dir", "convergence_results_full"
]

print("🚀 Running FULL convergence analysis on ALL 8 datasets...")
print(f"   Seeds: 10, Rounds: 50, Alpha: 0.3")
print(f"   Expected runtime: ~10-15 minutes\n")
print(f"Command: {' '.join(cmd)}\n")
print("="*80)

result = subprocess.run(cmd, cwd=".")
sys.exit(result.returncode)
