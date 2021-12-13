"""
Module for the trial evaluation callback class.
"""
from typing import Optional, Union

import optuna
import gym
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.monitor import Monitor


class TrialEvalCallback(EvalCallback):
    """
    stable_baselines3 callback used to evaluate and report a trial.
    """

    def __init__(self,
                 eval_env: Union[gym.Env, VecEnv, Monitor], trial: optuna.Trial,
                 callback_on_new_best: Optional[BaseCallback] = None, n_eval_episodes: int = 5,
                 eval_freq: int = 10000, log_path: Optional[str] = None,
                 best_model_save_path: Optional[str] = None, deterministic: bool = True,
                 render: bool = False, verbose: int = 1, warn: bool = True
                 ):
        super().__init__(eval_env, callback_on_new_best=callback_on_new_best,
                         n_eval_episodes=n_eval_episodes, eval_freq=eval_freq, log_path=log_path,
                         best_model_save_path=best_model_save_path, deterministic=deterministic,
                         render=render, verbose=verbose, warn=warn)
        self.trial = trial
        self.eval_index = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super()._on_step()
            self.eval_index += 1
            self.trial.report(self.last_mean_reward, self.eval_index)
            # Prune trial if needed
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True
