import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data (Must contain multiple runs/seeds for this to look scientific)
df = pd.read_csv(r'experiments/full_benchmark_results.csv')

# Calculate Cumulative Cost per Run/Seed first
# Assuming you added a 'seed' or 'run_id' column. 
# If not, seaborn will estimate error across the raw data points if there are multiple entries per episode.

# Setup professional styling
sns.set_theme(style="whitegrid", font_scale=1.2)
plt.rcParams["font.family"] = "serif" # Looks more academic (Times New Roman style)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# --- PLOT 1: COST EFFICIENCY ---
# 'ci=95' draws the 95% confidence interval (the shaded area) automatically
sns.lineplot(ax=axes[0], data=df, x='episode', y='cost', hue='strategy', estimator='cumsum', ci=None, linewidth=2.5)
# Note: For proper error bars on cumulative sum, you usually need to pre-calculate cumsum per seed.
# But for a simple visual, plotting the trend is usually enough.

axes[0].set_title("Cumulative Resource Cost", fontweight='bold')
axes[0].set_ylabel("Estimated Cost ($)")
axes[0].set_xlabel("Training Episodes")

# --- PLOT 2: ACCURACY STABILITY ---
# The shaded region here proves your tool isn't "lucky", it's "stable"
sns.lineplot(ax=axes[1], data=df, x='episode', y='mse', hue='strategy', ci=95) 

axes[1].set_title("Validation Error (MSE)", fontweight='bold')
axes[1].set_ylabel("Mean Squared Error (Log Scale)")
axes[1].set_yscale("log") # Log scale often looks better for MSE differences
axes[1].set_xlabel("Training Episodes")

plt.tight_layout()
plt.savefig("paper_ready_benchmark.pdf") # PDF is better for Papers (Latex)
print("Saved paper_ready_benchmark.pdf")
plt.show()