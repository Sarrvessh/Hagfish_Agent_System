import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Data
df = pd.read_csv(r'E:\Hagfish_Agent_System\experiments\experiments\full_benchmark_results.csv')

# Setup
sns.set_theme(style="whitegrid", palette="deep")
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# --- LEFT: Cumulative Cost ---
df['cumulative_cost'] = df.groupby('strategy')['cost'].cumsum()
sns.lineplot(ax=axes[0], data=df, x='episode', y='cumulative_cost', hue='strategy', linewidth=3)
axes[0].set_title("Total Cloud Cost (Cumulative)", fontsize=16, fontweight='bold')
axes[0].set_ylabel("Cost ($)")
axes[0].set_xlabel("Time (Episodes)")

# --- RIGHT: Metric Stability (Rolling Average) ---
# Smoothing MSE to see trends clearly
df['smoothed_mse'] = df.groupby('strategy')['mse'].transform(lambda x: x.rolling(5, min_periods=1).mean())
sns.lineplot(ax=axes[1], data=df, x='episode', y='smoothed_mse', hue='strategy', linewidth=2.5, alpha=0.8)
axes[1].set_title("Model Error (MSE - Smoothed)", fontsize=16, fontweight='bold')
axes[1].set_ylabel("MSE (Lower is Better)")
axes[1].set_xlabel("Time (Episodes)")

plt.tight_layout()
plt.savefig("benchmark_comparison_full.png")
print("Graph saved as benchmark_comparison_full.png")
plt.show()