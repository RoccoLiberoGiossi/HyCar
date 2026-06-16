import os
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy

from lib.RLagent.agent import HybridVehicleEnv

# 1. Setup directories for tracking logs and best models
log_dir = "./ppo_logs/"
os.makedirs(log_dir, exist_ok=True)

# 2. Create 16 parallel environments at once
# The monitor_dir parameter automatically handles wrapping each env in a Monitor
env = make_vec_env(HybridVehicleEnv, n_envs=16, monitor_dir=log_dir)

# 3. Create the Evaluation Callback
# This takes your 16-channel vector env and evaluates it deterministically
eval_callback = EvalCallback(
    env,
    best_model_save_path=log_dir,
    log_path=log_dir,
    eval_freq=50000,  # Evaluates every 50,000 steps TOTAL across all 16 channels
    deterministic=True,
    render=False,
)

# 4. Instantiate PPO exactly as before
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    gamma=0.95,
    n_steps=1024,
    batch_size=256,
    ent_coef=0.01,
    n_epochs=10,
    device="cuda",
)

# 5. Pass the callback into your massive 100M timestep matrix loop
model.learn(total_timesteps=50_000_000, callback=eval_callback, progress_bar=True)
# model.learn(total_timesteps=50_000_000, callback=eval_callback, progress_bar=True)

env.close()
