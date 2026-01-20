"""
Quick Convergence Demo - Test on 1 dataset (australian) before full run.

USAGE:
    python convergence_demo.py
"""

import subprocess
import sys

# Quick test: 1 dataset, 5 seeds, 50 rounds
cmd = [
    sys.executable,
    "convergence_analysis.py",
    "--seeds", "5",
    "--rounds", "50",
    "--alpha", "0.3",
    "--datasets", "australian",
    "--output-dir", "convergence_results_demo"
]

print("🚀 Running quick convergence demo on 'australian' dataset...")
print(f"Command: {' '.join(cmd)}\n")

result = subprocess.run(cmd, cwd=".")
sys.exit(result.returncode)
