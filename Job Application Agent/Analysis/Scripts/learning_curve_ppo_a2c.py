import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def plot_learning_curve(path_ppo, path_a2c, output_path):

    ppo_df = pd.read_csv(path_ppo, comment="#", header=None)
    a2c_df = pd.read_csv(path_a2c, comment="#", header=None)

    # Assign column names    
    cols = ["r", "l", "t", "result", "match_ratio", "page"]
    ppo_df.columns = cols
    a2c_df.columns = cols

    # Convert columns to numerics
    for df in [ppo_df, a2c_df]:
        df["t"] = pd.to_numeric(df["t"], errors="coerce")
        df["r"] = pd.to_numeric(df["r"], errors="coerce")

    # Apply smoothing
    window = 200
    ppo_df["reward_smooth"] = ppo_df["r"].rolling(window=window, min_periods=1).mean()
    a2c_df["reward_smooth"] = a2c_df["r"].rolling(window=window, min_periods=1).mean()

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(ppo_df["t"], ppo_df["reward_smooth"], label="PPO (smoothed)", alpha=0.9)
    plt.plot(a2c_df["t"], a2c_df["reward_smooth"], label="A2C (smoothed)", alpha=0.9)
    plt.title("Learning Curve (Smoothed): Episode Reward vs Timesteps")
    plt.xlabel("Timesteps")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Plot saved to: {output_path}")

# Paths
base_dir = Path(__file__).resolve().parent.parent.parent
path_ppo = base_dir / "Agent" / "logs" / "ppo" / "full_webflow" / "pass_merged.csv"
path_a2c = base_dir / "Agent" / "logs" / "a2c" / "full_webflow" / "pass.csv"
output_path = base_dir / "Analysis" / "Images" / "learning_curve_ppo_a2c.png"

plot_learning_curve(path_ppo, path_a2c, output_path)
