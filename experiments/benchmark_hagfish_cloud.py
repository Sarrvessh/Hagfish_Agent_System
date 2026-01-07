"""Benchmark script: Hagfish vs Optuna vs Random Search vs Fixed.
CORRECTED VERSION: Fixed 'Free Lunch' bug and 'Stateless Retrain' logic.
"""

import argparse
import ast
import os
import random
import time
import sys
from dataclasses import dataclass
from typing import Tuple, Optional, Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Try importing Optuna
try:
    import optuna
    OPTUNA_AVAILABLE = True
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Warning: Optuna not installed. 'OptunaPolicy' will be skipped.")

from adaptive_trainer import AdaptiveTrainer

# ================= DATA PROCESSING =================
# (This section remains unchanged from your previous robust version)
def load_csv_files(data_dir: str) -> pd.DataFrame:
    if not os.path.isdir(data_dir): raise ValueError(f"data_dir does not exist: {data_dir}")
    csv_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.lower().endswith('.csv')]
    if not csv_files: raise ValueError(f"No CSV files found in {data_dir}")
    frames = []
    print(f"Loading {len(csv_files)} CSV files...")
    for f in sorted(csv_files):
        try: frames.append(pd.read_csv(f, low_memory=False))
        except Exception as e: print(f"Skipping bad file {f}: {e}")
    if not frames: raise ValueError("No valid CSV files could be loaded.")
    return pd.concat(frames, ignore_index=True)

def ensure_columns(df: pd.DataFrame) -> Tuple[str, str]:
    time_candidates = ['time', 'timestamp', 'start_time', 'startTime']
    time_col = next((c for c in time_candidates if c in df.columns), next((c for c in df.columns if 'time' in c.lower()), None))
    cpu_candidates = ['average_usage', 'cpu_usage', 'maximum_usage', 'cpu', 'cpus']
    cpu_col = next((c for c in cpu_candidates if c in df.columns), next((c for c in df.columns if 'usage' in c.lower() or 'cpu' in c.lower()), None))
    if not time_col or not cpu_col: raise ValueError("Could not auto-detect columns.")
    return time_col, cpu_col

def parse_cpu_value(val):
    if pd.isna(val): return np.nan
    if isinstance(val, (int, float)): return float(val)
    s = str(val).strip()
    if s.startswith('{') and 'cpus' in s:
        try:
            d = ast.literal_eval(s)
            if isinstance(d, dict): return float(d.get('cpus', d.get('cpu', np.nan)))
        except: pass
    try: return float(s)
    except: return np.nan

def aggregate_to_5min(df: pd.DataFrame, time_col: str, cpu_col: str) -> pd.Series:
    df = df.copy()
    raw_times = pd.to_numeric(df[time_col], errors='coerce')
    valid_times = raw_times[raw_times > 0]
    max_val = valid_times.max()
    unit = 's'
    if max_val > 5e14: unit = 'us'
    elif max_val > 5e11: unit = 'ms'
    df['dt'] = pd.to_datetime(raw_times, unit=unit, errors='coerce')
    df = df.dropna(subset=['dt']).set_index('dt').sort_index()
    cpu_series = df[cpu_col].apply(parse_cpu_value).dropna()
    try: resampled = cpu_series.resample('5T').mean()
    except: resampled = cpu_series[cpu_series.index > cpu_series.index[-1] - pd.Timedelta(days=7)].resample('5T').mean()
    return resampled.interpolate(method='linear', limit=3).dropna()

def create_lag_features(series: pd.Series, lags: int = 12) -> Tuple[pd.DataFrame, pd.Series]:
    if len(series) < lags + 13: raise ValueError("Dataset too small.")
    df = pd.DataFrame({'t': series})
    for i in range(1, lags + 1): df[f'lag_{i}'] = df['t'].shift(i)
    df['target'] = df['t'].shift(-12)
    df = df.dropna()
    return df.drop(columns=['t', 'target']), df['target']


# ================= CLOUD ENVIRONMENT (FIXED: STATEFUL) =================

@dataclass
class EvalResult:
    metric: float
    cost: float
    time: float

class CloudEnvironment:
    def __init__(self, price_per_core_per_sec: float = 0.0001):
        self.price = price_per_core_per_sec
        # FIX: Store the active model to allow "Reuse" without retraining
        self.active_model = None

    def train_new_model(self, X_train, y_train, cores: int, lookback_window: int, random_state: int):
        """Trains a FRESH model and updates self.active_model"""
        cores = max(1, int(cores))
        rows = int(lookback_window * 12)
        X_curr = X_train.iloc[-rows:] if rows < len(X_train) else X_train
        y_curr = y_train.iloc[-rows:] if rows < len(y_train) else y_train

        model = RandomForestRegressor(n_estimators=50, n_jobs=cores, random_state=random_state)
        
        t0 = time.time()
        model.fit(X_curr, y_curr)
        train_time = time.time() - t0
        
        # Store state
        self.active_model = model
        
        cost = cores * train_time * self.price
        return cost, train_time

    def evaluate_active_model(self, X_val, y_val):
        """Evaluates the CURRENTLY stored model on validation data."""
        if self.active_model is None:
            # Fallback: If agent tries to skip training on very first step, force a cheap train
            # or return a "bad" metric.
            return -100.0 # Dummy bad MSE

        preds = self.active_model.predict(X_val)
        mse = mean_squared_error(y_val, preds)
        return -mse


# ================= AGENT POLICIES =================
# (Policies remain mostly the same, just standardizing interface)

class BasePolicy:
    def plan(self, problem_size, episode) -> Dict: pass
    def observe(self, metric, cost): pass

class FixedPolicy(BasePolicy):
    def plan(self, problem_size, episode):
        return {"cores": 16, "lookback": 48, "retrain_every": 1}

class RandomPolicy(BasePolicy):
    def plan(self, problem_size, episode):
        return {
            "cores": random.randint(2, 16),
            "lookback": random.choice([12, 24, 48, 96, 168]),
            "retrain_every": random.randint(1, 5)
        }

class OptunaPolicy(BasePolicy):
    def __init__(self, alpha=0.001):
        if not OPTUNA_AVAILABLE: raise ImportError("Optuna not found")
        self.study = optuna.create_study(direction='minimize')
        self.last_trial = None
        self.alpha = alpha
    def plan(self, problem_size, episode):
        self.last_trial = self.study.ask()
        return {
            "cores": self.last_trial.suggest_int("cores", 2, 16),
            "lookback": self.last_trial.suggest_int("lookback", 12, 168),
            "retrain_every": self.last_trial.suggest_int("retrain_every", 1, 5)
        }
    def observe(self, metric, cost):
        loss = (-metric) + (self.alpha * cost)
        self.study.tell(self.last_trial, loss)

class HagfishPolicy(BasePolicy):
    def __init__(self, alpha=0.001):
        self.trainer = AdaptiveTrainer(alpha=alpha)
    def plan(self, problem_size, episode):
        plan = self.trainer.plan({"dataset_size": problem_size})
        return {
            "cores": int(np.clip(plan.get('pop_size', 4), 2, 16)),
            "lookback": int(np.clip(plan.get('max_iter', 48), 12, 168)),
            "retrain_every": int(np.clip(plan.get('elite_size', 1), 1, 5))
        }
    def observe(self, metric, cost):
        self.trainer.observe(metric=metric, cost=cost, params=None, episode=None, elapsed_time=0)


# ================= EXECUTION LOOP (FIXED LOGIC) =================

def run_experiment(X, y, strategy, env, rounds, seed):
    split_idx = int(len(X) * 0.8)
    X_train_full, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train_full, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    records = []
    
    # FIX: Track time since last training explicitly
    steps_since_training = 1000 # Start high to force training on step 1
    
    for ep in range(1, rounds + 1):
        # 1. Get Plan
        plan = strategy.plan(problem_size=X.shape[1], episode=ep)
        
        # 2. Decide: Retrain or Reuse?
        # Logic: If we haven't trained in 'retrain_every' steps, we MUST train.
        should_retrain = (steps_since_training >= plan['retrain_every'])
        
        # Also force train if no model exists (Episode 1)
        if env.active_model is None:
            should_retrain = True

        if should_retrain:
            # TRAIN NEW MODEL
            cost, _ = env.train_new_model(
                X_train_full, y_train_full, 
                cores=plan['cores'], 
                lookback_window=plan['lookback'], 
                random_state=seed+ep
            )
            actual_cost = cost
            steps_since_training = 0 # Reset counter
        else:
            # SKIP TRAINING (Reuse active model)
            actual_cost = 0.0
            steps_since_training += 1 # Increment counter
            
        # 3. Evaluate (Always evaluate the Active Model)
        metric = env.evaluate_active_model(X_test, y_test)
        
        # 4. Feedback
        strategy.observe(metric=metric, cost=actual_cost)
            
        records.append({
            'episode': ep,
            'strategy': strategy.__class__.__name__,
            'mse': -metric,
            'cost': actual_cost,
            'retrained': should_retrain
        })
        
    return pd.DataFrame(records)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--rounds', type=int, default=50)
    args = parser.parse_args()

    raw_df = load_csv_files(args.data_dir)
    time_col, cpu_col = ensure_columns(raw_df)
    ts_series = aggregate_to_5min(raw_df, time_col, cpu_col)
    X, y = create_lag_features(ts_series)
    print(f"Dataset: {len(X)} samples.")

    env = CloudEnvironment(price_per_core_per_sec=0.0005)
    
    strategies = [FixedPolicy(), RandomPolicy(), HagfishPolicy(alpha=0.5)]
    if OPTUNA_AVAILABLE: strategies.append(OptunaPolicy(alpha=0.5))

    results = []
    for strat in strategies:
        print(f"Running {strat.__class__.__name__}...")
        # Reset environment model for each strategy
        env.active_model = None 
        df = run_experiment(X, y, strat, env, args.rounds, seed=42)
        results.append(df)
        
    final_df = pd.concat(results)
    final_df.to_csv('experiments/full_benchmark_results.csv', index=False)
    
    print("\n=== FINAL SCOREBOARD ===")
    print(final_df.groupby('strategy').agg({'mse': 'mean', 'cost': 'sum'}))

if __name__ == "__main__":
    main()