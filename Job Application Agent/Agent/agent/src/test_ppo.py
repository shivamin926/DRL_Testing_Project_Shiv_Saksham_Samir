# file: evaluate_model.py
import os
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from agent.handler.data_loader import Applicant, ApplicantManager
from agent.envs.webflow_env import WebFlowEnv

# --- Paths ---
MODEL_PATH = "models/ppo_full/ppo_webflow_resume_80000_steps.zip"
LOG_DIR = "logs/ppo/eval"
os.makedirs(LOG_DIR, exist_ok=True)

# --- Load applicants ---
manager = ApplicantManager(Applicant)
manager.load_from_json("agent/test_data/full_applicants.json", key="applicants")

# Test a few applicants (you can adjust this slice)
test_applicants = manager.entries

# --- Load trained PPO model ---
model = PPO.load(MODEL_PATH, device="cpu")

# --- Evaluation ---
results = []

for i, applicant in enumerate(test_applicants):
    print(f"\nEvaluating Applicant #{i}: {getattr(applicant, 'first_name', '?')} {getattr(applicant, 'last_name', '?')}")

    # --- Create environment for each applicant ---
    def make_env():
        return Monitor(
            WebFlowEnv(), 
            filename=os.path.join(LOG_DIR, f"eval_applicant_{i}.csv"),
            info_keywords=("result", "match_ratio", "page"),
        )

    env = DummyVecEnv([make_env])

    obs = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward.item()  # ensure float

    # --- Extract info safely ---
    info = info[0] if isinstance(info, list) else info
    result = info.get("result", "unknown")
    match_ratio = info.get("match_ratio", 0.0)
    page = info.get("page", "unknown")

    results.append({
        "applicant_id": i,
        "first_name": getattr(applicant, "first_name", ""),
        "last_name": getattr(applicant, "last_name", ""),
        "total_reward": float(total_reward),
        "result": result,
        "match_ratio": match_ratio,
        "page": page,
    })

    print(f"Applicant {i}: {result.upper()} ({match_ratio:.1%}) | Reward={total_reward:.1f}")
    env.close()

# --- Save results ---
df = pd.DataFrame(results)
summary_path = os.path.join(LOG_DIR, "evaluation_summary.csv")
df.to_csv(summary_path, index=False)

# --- Print summary ---
print("\nEvaluation Summary:")
if not df.empty:
    print(df.groupby("result")["match_ratio"].agg(["count", "mean"]))
else:
    print("No evaluation data collected.")
print(f"\nResults saved to: {summary_path}")
