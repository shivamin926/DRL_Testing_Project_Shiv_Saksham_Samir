import argparse
import csv
import os
import sys
import numpy as np
from stable_baselines3 import PPO, A2C

# Add the root directory to Python path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from envs.doodle_jump_env import DoodleJumpEnv

ALGOS = {"ppo": PPO, "a2c": A2C}

def infer_algo_from_path(path: str) -> str:
    lower = path.lower()
    if "a2c" in lower:
        return "a2c"
    return "ppo"

def evaluate(model_path: str, algo: str, episodes: int, render: bool, persona: str, out_csv: str|None):
    env = DoodleJumpEnv(render_mode="human" if render else None, seed=123, reward_preset=persona)
    Model = ALGOS[algo]
    model = Model.load(model_path, device="cpu")

    rows = []
    returns, lengths, heights, platforms_landed, deaths = [], [], [], [], []

    for ep in range(episodes):
        obs, info = env.reset()
        done, trunc = False, False
        ep_return, ep_len = 0.0, 0
        best_height = 0.0
        ep_platforms, ep_death = 0, 0

        while not (done or trunc):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, trunc, info = env.step(int(action))
            ep_return += reward
            ep_len += 1
            best_height = min(best_height, info.get("max_height", best_height))
            ep_death = info.get("death", ep_death)
            ep_platforms = info.get("platforms", ep_platforms)

        returns.append(ep_return)
        lengths.append(ep_len)
        heights.append(best_height)
        platforms_landed.append(ep_platforms)
        deaths.append(ep_death)

        print(f"Episode {ep+1}: return={ep_return:.2f}, steps={ep_len}, best_height={-best_height:.1f}, platforms={ep_platforms}, death={ep_death}")
        rows.append(dict(
            episode=ep+1, return_=ep_return, steps=ep_len,
            best_height=-best_height, platforms=ep_platforms, death=ep_death,
            algo=algo, persona=persona, model_path=model_path
        ))

    env.close()
    print("\n=== Aggregate Metrics ===")
    print(f"Mean return: {np.mean(returns):.2f} ± {np.std(returns):.2f}")
    print(f"Mean steps: {np.mean(lengths):.1f}")
    print(f"Mean best height: {-np.mean(heights):.1f}")
    print(f"Mean platforms landed: {np.mean(platforms_landed):.1f}")
    print(f"Deaths: {sum(deaths)}/{episodes} episodes ({100*sum(deaths)/episodes:.1f}%)")

    if out_csv:
        fieldnames = list(rows[0].keys()) if rows else ["episode","return_","steps","best_height","platforms","death","algo","persona","model_path"]
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"[eval] wrote metrics -> {out_csv}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="models/ppo_survivor_final.zip")
    ap.add_argument("--algo", choices=ALGOS.keys(), help="If omitted, inferred from model filename")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--persona", choices=["survivor", "greedy", "hunter"], default="survivor")
    ap.add_argument("--out_csv", type=str, default="logs/eval_metrics.csv")
    args = ap.parse_args()

    algo = args.algo or infer_algo_from_path(args.model_path)
    evaluate(args.model_path, algo, args.episodes, args.render, args.persona, args.out_csv)

if __name__ == "__main__":
    main()
