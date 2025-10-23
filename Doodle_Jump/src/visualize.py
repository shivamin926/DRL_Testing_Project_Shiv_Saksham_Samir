import os
import sys
import argparse
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="models/ppo_survivor_final.zip")
    parser.add_argument("--algo", choices=ALGOS.keys(), help="If omitted, inferred from model filename")
    parser.add_argument("--persona", choices=["survivor", "greedy", "hunter"], default="survivor")
    parser.add_argument("--debug", action="store_true", help="Print first 60 steps for debugging")
    parser.add_argument("--seed", type=int, default=123, help="Seed for consistency")
    args = parser.parse_args()

    # Ensure a window pops up (unset headless)
    if os.environ.get("SDL_VIDEODRIVER") == "dummy":
        del os.environ["SDL_VIDEODRIVER"]

    algo = args.algo or infer_algo_from_path(args.model_path)
    Model = ALGOS[algo]

    env = DoodleJumpEnv(render_mode="human", seed=args.seed, reward_preset=args.persona)
    model = Model.load(args.model_path)

    obs, info = env.reset()
    done, trunc = False, False
    t = 0
    while not (done or trunc):
        action, _ = model.predict(obs, deterministic=False)
        obs, reward, done, trunc, info = env.step(int(action))
        if args.debug and t < 60:
            print(f"t={t:02d} a={int(action)} y={env.player.y:.1f} vy={env.player.vy:.2f} r={reward:.2f}")
        t += 1

    env.close()

if __name__ == "__main__":
    main()
