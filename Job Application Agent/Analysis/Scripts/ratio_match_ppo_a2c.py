import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def plot_match_ratio(path_ppo, path_a2c, output_path="match_ratio_plot.png"):
    ppo_df = pd.read_csv(path_ppo, comment="#", header=None)
    a2c_df = pd.read_csv(path_a2c, comment="#", header=None)

    # Assign column names    
    cols = ["r", "l", "t", "result", "match_ratio", "page"]
    ppo_df.columns = cols
    a2c_df.columns = cols

    # Convert columns to numeric
    for df in [ppo_df, a2c_df]:
        df["t"] = pd.to_numeric(df["t"], errors="coerce")
        df["match_ratio"] = pd.to_numeric(df["match_ratio"], errors="coerce")

    # Apply smoothing
    window = 200
    ppo_df["match_smooth"] = ppo_df["match_ratio"].rolling(window=window, min_periods=1).mean()
    a2c_df["match_smooth"] = a2c_df["match_ratio"].rolling(window=window, min_periods=1).mean()

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(ppo_df["t"], ppo_df["match_smooth"], label=f"PPO (avg over {window})", color="tab:blue", alpha=0.9)
    plt.plot(a2c_df["t"], a2c_df["match_smooth"], label=f"A2C (avg over {window})", color="tab:orange", alpha=0.9)
    plt.title("Match Ratio Over Time")
    plt.xlabel("Timesteps")
    plt.ylabel("Match Ratio (0–1, smoothed)")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Match ratio plot saved to: {output_path}")

# Paths
base_dir = Path(__file__).resolve().parent.parent.parent
path_ppo = base_dir / "Agent" / "logs" / "ppo" / "full_webflow" / "pass_merged.csv"
path_a2c = base_dir / "Agent" / "logs" / "a2c" / "full_webflow" / "pass.csv"
output_path = base_dir / "Analysis" / "Images" / "ratio_match_ppo_a2c.png"

plot_match_ratio(path_ppo, path_a2c, output_path)
