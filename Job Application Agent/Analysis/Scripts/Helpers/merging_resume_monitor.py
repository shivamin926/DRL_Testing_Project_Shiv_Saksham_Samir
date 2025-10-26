import pandas as pd
from pathlib import Path

# Set paths 
base_dir = Path(__file__).resolve().parent.parent.parent
old_path = base_dir / "Agent" / "logs" / "ppo" / "full_webflow" / "pass1.csv"
new_path = base_dir / "Agent" / "logs" / "ppo" / "full_webflow" / "pass2.csv"
merged_path = base_dir / "Agent" / "logs" / "ppo" / "full_webflow" / "pass_merged.csv"

with open(old_path, "r") as f:
    header_line = f.readline().strip()
    print(f"Header line preserved: {header_line}")

old_df = pd.read_csv(old_path, comment="#", header=None)
new_df = pd.read_csv(new_path, comment="#", header=None)

cols = ["r", "l", "t", "result", "match_ratio", "page"]
old_df.columns = cols
new_df.columns = cols

for c in ["r", "l", "t", "match_ratio"]:
    old_df[c] = pd.to_numeric(old_df[c], errors="coerce")
    new_df[c] = pd.to_numeric(new_df[c], errors="coerce")

old_df.dropna(subset=["t", "r"], inplace=True)
new_df.dropna(subset=["t", "r"], inplace=True)

# Find last timestep from old data
last_timestep = old_df["t"].iloc[-1]
print(f"Last timestep in old log: {last_timestep}")

# Offset new timesteps
new_df["t"] = new_df["t"] + last_timestep

# Merge
merged_df = pd.concat([old_df, new_df], ignore_index=True)

with open(merged_path, "w") as f:
    f.write(f"{header_line}\n")
    f.write("r,l,t,result,match_ratio,page\n")
merged_df.to_csv(merged_path, mode="a", index=False, header=False)

print(f"✅ Merged file saved to: {merged_path}")
