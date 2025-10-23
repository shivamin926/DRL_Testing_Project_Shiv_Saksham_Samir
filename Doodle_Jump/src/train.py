import os
import sys
import argparse
import torch
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.logger import configure

# Add the root directory to Python path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from envs.doodle_jump_env import DoodleJumpEnv

ALGOS = {"ppo": PPO, "a2c": A2C}

LOG_DIR = "logs"
MODEL_DIR = "models"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def make_env(render_mode=None, seed=0, persona="survivor"):
    def _thunk():
        env = DoodleJumpEnv(render_mode=render_mode, seed=seed, reward_preset=persona)
        return env
    return _thunk

def train_one(algo_name: str, total_timesteps: int, seed: int, persona: str, tag: str):
    Model = ALGOS[algo_name]

    run_name = f"{algo_name}_{persona}{('_' + tag) if tag else ''}"
    monitor_csv = os.path.join(LOG_DIR, f"{run_name}_monitor.csv")

    env = DummyVecEnv([make_env(None, seed, persona)])
    env = VecMonitor(env, filename=monitor_csv)

    eval_env = DummyVecEnv([make_env(None, seed + 1, persona)])
    eval_env = VecMonitor(eval_env)

    logger = configure(os.path.join(LOG_DIR, run_name), ["stdout", "csv", "tensorboard"])

    model = Model(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=os.path.join(LOG_DIR, "tb"),
        device="cuda" if torch.cuda.is_available() else "cpu",
        learning_rate=2.5e-4,
        ent_coef=0.10,          
        vf_coef=0.5,
        gamma=0.995 if algo_name == "ppo" else 0.99,
        gae_lambda=0.95 if algo_name == "ppo" else 1.0,
        n_steps=2048 if algo_name == "ppo" else 5,
        seed=seed,
        policy_kwargs=dict(
            net_arch=[dict(pi=[128,128,64], vf=[128,128,64])],
            activation_fn=torch.nn.ReLU,
            ortho_init=True,
        ),
    )
    model.set_logger(logger)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=MODEL_DIR,
        log_path=LOG_DIR,
        eval_freq=25_000,
        deterministic=True,
        render=False,
        n_eval_episodes=10,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=100_000,
        save_path=MODEL_DIR,
        name_prefix=f"{run_name}",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    print(f"[train] run={run_name} timesteps={total_timesteps} seed={seed}")
    model.learn(total_timesteps=total_timesteps, callback=[eval_cb, ckpt_cb])
    final_path = os.path.join(MODEL_DIR, f"{run_name}_final")
    model.save(final_path)
    print(f"[train] saved -> {final_path}.zip")

    env.close()
    eval_env.close()

def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--algo", choices=ALGOS.keys(), help="Train a single algorithm.")
    g.add_argument("--both", action="store_true", help="Train PPO and A2C sequentially.")
    p.add_argument("--persona", choices=["survivor", "greedy", "hunter"], default="survivor")
    p.add_argument("--seed", type=int, default=123, help="Training seed")
    p.add_argument("--steps", type=int, default=3_000_000, help="Total timesteps per run")
    p.add_argument("--tag", type=str, default="", help="Optional label for this run")
    args = p.parse_args()

    if args.both:
        for a in ["ppo", "a2c"]:
            train_one(a, total_timesteps=args.steps, seed=args.seed, persona=args.persona, tag=args.tag)
    else:
        train_one(args.algo, total_timesteps=args.steps, seed=args.seed, persona=args.persona, tag=args.tag)

if __name__ == "__main__":
    main()
