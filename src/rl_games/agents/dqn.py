from pathlib import Path
from typing import Self

import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy


class DQNAgent:
    def __init__(self, env_id: str, **kwargs) -> None:
        self.env_id = env_id
        self.model = DQN(
            "MlpPolicy",
            env_id,
            verbose=1,
            exploration_final_eps=0.1,
            target_update_interval=250,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # core RL
    # ------------------------------------------------------------------

    def train(self, total_timesteps: int = 100_000) -> None:
        self.model.learn(total_timesteps=total_timesteps)

    def predict(
        self, obs: np.ndarray, *, deterministic: bool = True
    ) -> tuple[np.ndarray, None]:
        return self.model.predict(obs, deterministic=deterministic)

    def evaluate(self, n_episodes: int = 10) -> tuple[float, float]:
        eval_env = gym.make(self.env_id)
        mean_reward, std_reward = evaluate_policy(
            self.model, eval_env, n_eval_episodes=n_episodes, deterministic=True
        )
        eval_env.close()
        return float(mean_reward), float(std_reward)

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))
        print(f"Saved DQN agent to {path}")

    @classmethod
    def load(cls, path: Path, env_id: str) -> Self:
        agent = object.__new__(cls)
        agent.env_id = env_id
        agent.model = DQN.load(str(path))
        return agent

    def set_env(self, env_id: str) -> None:
        """Attach a fresh training environment (needed for continued training)."""
        self.model.set_env(gym.make(env_id))

    def info(self) -> str:
        ts = self.model.num_timesteps
        return (
            f"DQN agent for {self.env_id}\n"
            f"  Timesteps trained : {ts}\n"
            f"  Policy            : {self.model.policy.__class__.__name__}\n"
            f"  Exploration rate  : {self.model.exploration_rate:.4f}"
        )
