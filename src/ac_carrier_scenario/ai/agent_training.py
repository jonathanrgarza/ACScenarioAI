"""
Module for training the AI agent.
"""
import json
import logging
import os
import time
from datetime import timedelta
from pathlib import Path
from pprint import pformat
from typing import Optional, Union, Any, Type, Callable

import gym
import numpy as np
import optuna
from gym import Env
from gym.wrappers import TimeLimit
from optuna.visualization import plot_optimization_history, plot_param_importances
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import Logger, configure, INFO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv

from torch import nn

# Import the custom env
# noinspection PyUnresolvedReferences
from ac_carrier_scenario.common.scenarios import AircraftCarrierScenario
from ac_carrier_scenario.common.environment import SpecificAircraftCarrierScenarioEnv
import ac_carrier_scenario.util.stay_awake as stay_awake
from ac_carrier_scenario.ai.callbacks import TrialEvalCallback, TrainingEvalCallback


def clear_console() -> None:
    """
    Clears the console's display.

    :return: None
    """
    import os
    clear_command: str = "cls"
    if os.name != "nt":
        clear_command = "clear"
    os.system(clear_command)


def factors(n: int) -> list[int]:
    """
    Gets an ordered list of the factors for a given number.

    :param n: The number to get factors for.
    :return: A list of all the factors for a given number.
    :rtype: list
    """
    factor_list = []
    for i in range(1, int(n ** 0.5) + 1):
        if n % i == 0:
            factor_list.append(i)
            factor_list.append(n // i)

    factor_list = list(dict.fromkeys(factor_list))  # Get only unique entries
    factor_list.sort()
    return factor_list


def get_ideal_scenario_env() -> Env:
    """
    Gets the ideal scenario gym environment.
    :return: The ideal scenario environment.
    """
    scenario = AircraftCarrierScenario(missile_count=13, jet_count=19, pilot_count=21,
                                       target1_expected_damage=4, target2_expected_damage=2,
                                       target3_expected_damage=2, target4_expected_damage=2,
                                       target5_expected_damage=0, target6_expected_damage=0)

    env: SpecificAircraftCarrierScenarioEnv = SpecificAircraftCarrierScenarioEnv(scenario=scenario)
    wrapped_env: Env = TimeLimit(env, max_episode_steps=250)
    return wrapped_env


_optuna_logger: logging.Logger  # Logger for optuna studies


def net_arch_to_dict(size: str) -> list[dict[str, list[int]]]:
    """
    Converts a net_arch string to the correct type. Only useful for PPO or A2C agents.
    :param size: The size string to convert. Supports: 'small' or 'medium'
    :return: The converted value.
    """
    lowered_size: str = size.lower()
    if lowered_size == "small":
        return [dict(pi=[64, 64], vf=[64, 64])]
    elif lowered_size == "medium":
        return [dict(pi=[256, 256], vf=[256, 256])]
    raise ValueError("Invalid size string. Only 'small' or 'medium' is supported.")


def activation_fn_to_type(name: str) -> Type[Union[nn.Tanh, nn.ReLU, nn.ELU, nn.LeakyReLU]]:
    """
    Converts a net_arch string to the correct type.
    :param name: The name string to convert. Supports: 'tanh', 'relu', 'elu', or 'leaky_relu'
    :return: The converted value (a type).
    """
    lowered_name: str = name.lower()
    if lowered_name == "tanh":
        return nn.Tanh
    elif lowered_name == "relu":
        return nn.ReLU
    elif lowered_name == "elu":
        return nn.ELU
    elif lowered_name == "leaky_relu":
        return nn.LeakyReLU
    raise ValueError("Invalid name string. Only 'tanh', 'relu', 'elu', or 'leaky_relu' is supported.")


optimization_tb_log_path: Optional[str] = None  # "models/optuna/logging"
"""
    The tensorboard log path to use during optimization.
"""

optimization_total_timesteps: int = 25000  # "25000"
"""
    The total number of timesteps to use for each trial.
"""

optimization_n_eval_episodes: int = 5  # "5"
"""
    The number of episodes used for evaluation for each trial. Must be greater than 1.
"""

optimization_eval_freq: int = 10000  # "10000"
"""
    The timesteps between evaluations for each trial. Must be less than optimization_total_timesteps but greater than 1.
"""

optimization_n_envs: int = 1  # "1"
"""
    The number of parallel environments to use during optimization.
"""

optimization_learning_verbose: int = 0  # "0" : No output
"""
    If the optimization learning will have verbose output.
"""

optimization_eval_verbose: int = 1  # "0" : No output
"""
    If the optimization learning evaluation will have verbose output.
"""


def agent_objective(trial: optuna.Trial) -> int:
    """
    Function to use with optuna to tune hyperparameters for ACS environment.

    :param trial: The optuna trial
    :return: The mean reward for the trial
    :rtype: int
    """
    global _optuna_logger
    global optimization_tb_log_path
    global optimization_total_timesteps
    global optimization_n_eval_episodes
    global optimization_eval_freq
    global optimization_n_envs
    global optimization_learning_verbose
    global optimization_eval_verbose

    # Create the gym model
    # gym_env: Env = gym.make("ACS-v0")
    monitored_env = make_vec_env("ACS-v0", n_envs=optimization_n_envs)

    # eval_env = Monitor(gym.make("ACS-v0"))
    eval_env = Monitor(get_ideal_scenario_env())

    # Determine the hyperparameters
    # algorithm = trial.suggest_categorical("algorithm", ["PPO", "A2C"])
    algorithm = "PPO"
    # policy = "MlpPolicy"
    policy = "MultiInputPolicy"

    # Get trial's hyperparameters that are common to all algorithms
    lr_schedule = "constant"
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1)
    n_steps = trial.suggest_categorical("n_steps", [8, 16, 32, 64, 128, 256, 512, 1024, 2048])
    gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    ent_coef = trial.suggest_loguniform("ent_coef", 0.00000001, 0.1)
    vf_coef = trial.suggest_uniform("vf_coef", 0, 1)
    gae_lambda = trial.suggest_categorical("gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])
    max_grad_norm = trial.suggest_categorical("max_grad_norm", [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5])

    net_arch = trial.suggest_categorical("net_arch", ["small", "medium"])
    activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu"])

    # Convert net_arch and activation_fn to correct value/types
    net_arch = net_arch_to_dict(net_arch)
    activation_fn = activation_fn_to_type(activation_fn)

    # Launch tensorboard with command: tensorboard --logdir=models/optuna/logging

    if algorithm == "PPO":
        # Get trial's hyperparameters that are for PPO algorithm only
        clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3, 0.4])
        n_epochs = trial.suggest_categorical("n_epochs", [1, 5, 10, 20])
        ortho_init = False

        # Convert learning_rate to a linear schedule IF schedule is linear
        if lr_schedule == "linear":
            learning_rate = linear_schedule(learning_rate)

        # Suggestion: factors of n_steps * n_envs (number of environments (parallel))
        # batch_size = trial.suggest_categorical("batch_size", factors(n_steps))
        # n_steps_factors = factors(n_steps)
        # n_steps_factors_copy = n_steps_factors.copy()  # Copy for debugging
        # if len(n_steps_factors) > 2:
        #     n_steps_factors.pop()
        # batch_size = n_steps_factors.pop()  # Get second-largest factor (or last factor if only two)
        #
        # if batch_size == 1:
        #     print("Invalid batch_size would have been picked")
        #     print(f"Factors of {n_steps} were: {n_steps_factors_copy}")
        #     batch_size = n_steps

        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128, 256, 512])

        if batch_size > n_steps:
            batch_size = n_steps

        policy_kwargs: dict[str, Any] = {
            "net_arch": net_arch,
            "activation_fn": activation_fn,
            "ortho_init": ortho_init
        }

        model = PPO(policy, monitored_env, learning_rate=learning_rate, gamma=gamma, n_steps=n_steps,
                    n_epochs=n_epochs, batch_size=batch_size, ent_coef=ent_coef, clip_range=clip_range,
                    gae_lambda=gae_lambda, max_grad_norm=max_grad_norm, vf_coef=vf_coef,
                    policy_kwargs=policy_kwargs,
                    tensorboard_log=optimization_tb_log_path, verbose=optimization_learning_verbose)
    elif algorithm == "A2C":
        # Get trial's hyperparameters that are for A2C algorithm only
        ortho_init = trial.suggest_categorical("ortho_init", [False, True])
        normalize_advantage = trial.suggest_categorical("normalize_advantage", [False, True])
        # Toggle PyTorch RMS Prop (different from TF one, cf doc)
        use_rms_prop = trial.suggest_categorical("use_rms_prop", [False, True])

        lr_schedule = trial.suggest_categorical("lr_schedule", ["linear", "constant"])

        # Convert learning_rate to a linear schedule IF schedule is linear
        if lr_schedule == "linear":
            learning_rate = linear_schedule(learning_rate)

        policy_kwargs: dict[str, Any] = {
            "net_arch": net_arch,
            "activation_fn": activation_fn,
            "ortho_init": ortho_init
        }

        model = A2C(policy, monitored_env, learning_rate=learning_rate, n_steps=n_steps, gamma=gamma,
                    gae_lambda=gae_lambda, ent_coef=ent_coef, vf_coef=vf_coef, max_grad_norm=max_grad_norm,
                    use_rms_prop=use_rms_prop, normalize_advantage=normalize_advantage,
                    policy_kwargs=policy_kwargs,
                    tensorboard_log=optimization_tb_log_path, verbose=optimization_learning_verbose)
    else:
        raise ValueError(f"Invalid algorithm selected: {algorithm}")

    # Create the evaluation callback
    eval_callback = TrialEvalCallback(eval_env, trial, n_eval_episodes=optimization_n_eval_episodes,
                                      eval_freq=max(optimization_eval_freq // optimization_n_envs, 1),
                                      verbose=optimization_eval_verbose)

    try:
        # No keep awake needed here as this is called by optuna which has been kept awake
        _optuna_logger.info(f"Starting a trial '{trial.number}' using the '{algorithm}' algorithm")
        model.learn(optimization_total_timesteps, callback=eval_callback)

        model.env.close()
        eval_env.close()
    except (AssertionError, ValueError) as e:
        # Sometimes, random hyperparameters can generate NaN
        model.env.close()
        eval_env.close()
        # Prune hyperparameters that generate NaNs
        _optuna_logger.info(e)
        _optuna_logger.info("============")
        _optuna_logger.info("Sampled hyperparameters:")
        _optuna_logger.info(trial.params)
        raise optuna.exceptions.TrialPruned()

    is_pruned = eval_callback.is_pruned
    reward = eval_callback.last_mean_reward

    del model.env, eval_callback
    del model

    if is_pruned:
        raise optuna.exceptions.TrialPruned()

    return reward


def perform_optuna_optimizing(study_name: str = "agent_study_2", n_trials: int = 100, total_timesteps: int = 25000,
                              n_eval_episodes: int = 5, eval_freq: int = 10000, n_envs: int = 1,
                              learning_verbose_level: int = 0, eval_verbose_level: int = 1,
                              tb_log_path: Optional[str] = None):
    global _optuna_logger
    global optimization_tb_log_path
    global optimization_total_timesteps
    global optimization_n_eval_episodes
    global optimization_eval_freq
    global optimization_n_envs
    global optimization_learning_verbose
    global optimization_eval_verbose
    # Add stream handler of stdout to show the messages
    _optuna_logger = optuna.logging.get_logger("optuna")
    # These are not needed as they are the default for optuna's logger
    # log_handler = logging.StreamHandler(sys.stdout)
    # log_handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s"))
    # logger.addHandler(log_handler)

    if not isinstance(study_name, str):
        raise ValueError(f"study_name is not of type 'str': {type(study_name)}")

    _optuna_logger.info("Initializing an optuna hyperparameter optimization study run")

    # Create dir if needed
    try:
        os.makedirs("models/optuna", exist_ok=True)
    except OSError:
        _optuna_logger.error("Could not create folder 'models/optuna'. Create this folder first and try again.")
        exit(1)

    storage_name = f"sqlite:///models/optuna/{study_name}.db"
    study = optuna.create_study(study_name=study_name, storage=storage_name, direction="maximize", load_if_exists=True)
    optimization_tb_log_path = tb_log_path

    optimization_total_timesteps = total_timesteps
    optimization_n_eval_episodes = n_eval_episodes
    optimization_eval_freq = eval_freq
    optimization_n_envs = n_envs
    optimization_learning_verbose = learning_verbose_level
    optimization_eval_verbose = eval_verbose_level

    if optimization_n_eval_episodes <= 0:
        raise ValueError("n_eval_episodes must be greater than 0")

    if optimization_eval_freq > optimization_total_timesteps:
        raise ValueError("eval_freq cannot be greater than total_timesteps")

    completed = False

    try:
        with stay_awake.keep_awake():
            _optuna_logger.info("Starting an optuna hyperparameter optimization run")
            _optuna_logger.info(f"Optimization run parameters: Total Timesteps: {optimization_total_timesteps}, "
                                f"Eval Episodes: {optimization_n_eval_episodes}, Eval Freq: {optimization_eval_freq}, "
                                f"Number of Environments: {optimization_n_envs}")
            study.optimize(agent_objective, n_trials=n_trials)
            completed = True
    except KeyboardInterrupt:
        pass

    try:
        trial = study.best_trial
    except ValueError:
        _optuna_logger.warning("Could not get a best trial. Likely because no trial has ever been completed yet")
        return

    _optuna_logger.info(f"Number of finished trials: {len(study.trials)}")

    _optuna_logger.info(f"Best trial: {trial.number}")

    _optuna_logger.info(f"Value: {trial.value}")

    ideal_params = trial.params
    _optuna_logger.info("Params: ")
    for key, value in ideal_params.items():
        _optuna_logger.info(f"    {key}: {value}")

    if completed:
        completed_str = "Completed"
    else:
        completed_str = "Uncompleted"

    report_name = (
        f"report_study-{study_name}_{n_trials}-trials-{optimization_total_timesteps}"
        f"-TPE-None_{int(time.time())}-status-{completed_str}.csv"
    )

    log_path = os.path.join("models", "optuna", report_name)

    _optuna_logger.info(f"Writing report to {log_path}")

    # Write report
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    study.trials_dataframe().to_csv(log_path)

    # Write best params
    report_name = (
        f"ideal_params_study-{study_name}_{n_trials}-trials-{optimization_total_timesteps}"
        f"-TPE-None_{int(time.time())}-status-{completed_str}.json"
    )

    log_path = os.path.join("models", "optuna", report_name)

    _optuna_logger.info(f"Writing ideal parameters to {log_path}")
    with open(log_path, "w") as json_file:
        json.dump(obj=ideal_params, fp=json_file, indent=4)

    # Plot optimization result
    _optuna_logger.info("Displaying optimization plots")
    try:
        fig1 = plot_optimization_history(study)
        fig2 = plot_param_importances(study)

        fig1.show()
        fig2.show()
    except (ValueError, ImportError, RuntimeError) as e:
        _optuna_logger.warning(f"Could not plot study: {e}")


def test_agent(model, env: Union[Env, Monitor, VecEnv], n_eval_episodes: int = 10):
    if isinstance(env, Monitor) or isinstance(env, VecEnv):
        monitored_env = env
    else:
        monitored_env = Monitor(env)

    model.set_env(monitored_env)
    return evaluate_policy(model, monitored_env, n_eval_episodes=n_eval_episodes)


def linear_schedule(initial_value: Union[float, str]) -> Callable[[float], float]:
    """
    Linear learning rate schedule.
    :param initial_value: The initial value (float or str)
    :return: The delegate function
    """
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    def linear_algo(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress_remaining: (float)
        :return: (float)
        """
        return progress_remaining * initial_value

    return linear_algo


def get_agent_hyperparameters_raw() -> dict[str, Union[str, int, dict[str, Any]]]:
    # Set the best hyperparams found; value reached for these parameters: -18
    net_arch = "small"
    activation_fn = "relu"
    ortho_init = False

    policy_kwargs: dict[str, Union[str, bool]] = {
        "net_arch": net_arch,
        "activation_fn": activation_fn,
        "ortho_init": ortho_init
    }

    model_parameters = {
        "policy": "MultiInputPolicy",  # This policy is required for Dict type of environments
        # Hyperparameters
        "batch_size": 8,
        "clip_range": 0.4,
        "ent_coef": 2.931484907646462e-05,
        "gae_lambda": 1.0,
        "gamma": 0.9,
        "learning_rate": 0.0001271857047377097,
        "max_grad_norm": 5,
        "n_epochs": 20,
        "n_steps": 128,
        "vf_coef": 0.06252372815378887,
        "policy_kwargs": policy_kwargs
    }

    return model_parameters


def get_agent_hyperparameters() -> dict[str, Any]:
    model_parameters = get_agent_hyperparameters_raw()

    policy_kwargs = model_parameters["policy_kwargs"]

    policy_kwargs["net_arch"] = net_arch_to_dict(
        policy_kwargs["net_arch"])
    policy_kwargs["activation_fn"] = activation_fn_to_type(
        policy_kwargs["activation_fn"])

    return model_parameters


def get_new_ppo_agent(env: Union[Env, VecEnv, str], logger: Optional[Logger] = None,
                      tensorboard_log: Optional[str] = None, verbose: int = 0) -> PPO:
    """
    Gets a new PPO agent for the given environment. This is meant only for the AC Carrier Scenario.

    :param env: The gym environment.
    :param logger: The logger to use. Default is None.
    :param tensorboard_log: The tensorboard log location. Default is None.
    :param verbose: The verbose level. 0 = None, 1 = info and 2 = debug. Default is 0.
    :return: The PPO agent.
    """
    if env is None:
        raise ValueError("The env is not set")

    model_parameters = get_agent_hyperparameters()

    if logger is not None and logger.level >= INFO:
        # Print out more user-friendly version of the hyperparameters
        logger.info("Hyperparameters for agent")
        logger.info(pformat(get_agent_hyperparameters_raw()))

    model = PPO(env=env, tensorboard_log=tensorboard_log, verbose=verbose, **model_parameters)
    return model


def perform_agent_training(logger: Logger, model_save_path: str = "models/trained_model_v2",
                           best_model_save_path: str = "models", tb_log_path: Optional[str] = None,
                           n_envs: int = 6, total_timesteps: int = 25000, n_eval_episodes: int = 5,
                           eval_freq: int = 10000, learning_verbose: int = 0, eval_verbose: int = 0):
    # Parallel Environments
    env = make_vec_env("ACS-v0", n_envs=n_envs)

    # Convert str to a Path instance
    path = Path(model_save_path)
    if path.suffix.lower() != ".zip" and path.suffix != "":
        raise ValueError("The model_save_path must be a valid path to a .zip file "
                         "OR valid file path without the .zip extension")

    model = None
    if path.exists() or path.suffix == "" or path.suffix.lower() == ".zip":
        try:
            model = PPO.load(path, env, tensorboard_log=tb_log_path)
            model.verbose = learning_verbose  # Update verbose level for loaded model
        except FileNotFoundError:
            model = None

    if path.suffix == "":
        path = Path(path.parent, f"{path.stem}.zip")

    if model is None:
        logger.info(f"No existing model found at '{path}'. Creating a new model to learn with")
        model = get_new_ppo_agent(env=env, logger=logger, tensorboard_log=tb_log_path, verbose=learning_verbose)
    else:
        logger.info(f"Existing model found at '{path}'. Will continue its learning")

    if learning_verbose > 0:
        model.set_logger(logger)

    # Set callbacks
    # Separate evaluation environment
    eval_env = Monitor(gym.make("ACS-v0"))
    # callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=10, verbose=1)
    callback_on_best = None
    eval_callback = TrainingEvalCallback(eval_env=eval_env, callback_on_new_best=callback_on_best,
                                         best_model_save_path=best_model_save_path, eval_freq=eval_freq,
                                         n_eval_episodes=n_eval_episodes, verbose=eval_verbose)
    with stay_awake.keep_awake():
        try:
            model.learn(total_timesteps=total_timesteps, callback=eval_callback)
        except KeyboardInterrupt:
            pass

    model.save(path)
    env.close()
    logger.info(f"Training complete. Results saved to: '{path}'")

    return model


def _perform_agent_run(logger: Logger, env: Monitor, model, verbose: bool = False):
    state = env.reset()
    done = False

    reward_score = 0
    steps = 0
    info = dict()
    for _ in range(251):
        action, _states = model.predict(state)
        new_state, reward, done, info = env.step(action)
        state = new_state

        reward_score += reward
        steps += 1

        # clear_console()
        # env.render()
        if verbose:
            logger.info(f"Step#: {steps}, Reward: {reward}, Action: {action}, currentShipDamage: "
                        f"{state['currentShipDamage']}, missiles: {state['missiles']}")
        if verbose:
            time.sleep(0.3)

        if done:
            if verbose:
                logger.info(f"Reward Sum/Score: {reward_score}, Steps: {steps}")
            break

    completed = True
    if not done or "TimeLimit.truncated" in info:
        completed = False

    if not completed and verbose:
        logger.info("Agent could not complete the environment")

    ideal_env: Union[DummyVecEnv, VecEnv, SpecificAircraftCarrierScenarioEnv] = env.unwrapped
    if isinstance(ideal_env, DummyVecEnv):
        ideal_env = ideal_env.envs[0]

    if verbose:
        if ideal_env.is_expected_damage_met:
            logger.info("Agent met expected damage requirements")
        else:
            logger.info("Agent FAILED to meet expected damage requirements")
            logger.info(f"Env: {state}")
    return reward_score, steps, completed, ideal_env.is_expected_damage_met


def _perform_agent_runs(logger: Logger, model, n_episodes: int = 1, use_ideal_scenario: bool = True):
    logger.info(
        f"Starting a run of {n_episodes} episodes using a {'ideal' if use_ideal_scenario else 'random'} environment")
    if use_ideal_scenario:
        env = Monitor(get_ideal_scenario_env())
    else:
        env = make_vec_env("ACS-v0")
    model.set_env(env)

    if n_episodes == 1:
        _perform_agent_run(logger=logger, env=env, model=model, verbose=True)
    else:
        total_reward = []
        total_steps = []
        complete_envs = []
        expected_damage_met_list = []
        for _ in range(n_episodes):
            reward_score, steps, done, expected_damage_met = _perform_agent_run(logger=logger, env=env,
                                                                                model=model, verbose=False)
            total_reward.append(reward_score)
            total_steps.append(steps)
            complete_envs.append(done)
            expected_damage_met_list.append(expected_damage_met)

        reward_avg = np.mean(total_reward)
        reward_std = np.std(total_reward)
        steps_avg = np.mean(total_steps)
        steps_std = np.std(total_steps)
        done_avg = np.mean(complete_envs) * 100
        expected_damage_met_avg = np.mean(expected_damage_met_list) * 100

        logger.info(f"Avg Reward Score: {reward_avg:.0f} +/- {reward_std:.0f}, "
                    f"Avg Steps: {steps_avg:.0f} +/- {steps_std:.0f}, "
                    f"Completion Avg: {done_avg:.1f}%, Expected Damage Met Avg: {expected_damage_met_avg:.1f}%")

    env.close()
    logger.info("Run complete")


def train_agent(perform_training: bool, perform_test: bool, run_env: bool,
                use_best_model: bool = True,
                model_save_path: str = "models/trained_model_v2",
                best_model_save_path: str = "models", tb_log_path: Optional[str] = None,
                n_envs: int = 6, total_timesteps: int = 25000, n_eval_episodes: int = 5, eval_freq: int = 10000,
                learning_verbose_level: int = 0, eval_verbose_level: int = 1, test_n_eval_episodes: int = 10):
    # Init logger
    logger: Logger = configure(folder=None, format_strings=["stdout"])

    logger.info("Running stable_baselines3 PPO agent training")

    if not perform_training and not perform_test and not run_env:
        logger.warn("No action to perform, please select at least one action and run again")
        return

    if perform_training:
        model = perform_agent_training(logger, model_save_path, best_model_save_path,
                                       tb_log_path, n_envs, total_timesteps, n_eval_episodes, eval_freq,
                                       learning_verbose_level, eval_verbose_level)
    else:
        logger.info("Training option disabled. Loading model from file")
        env = make_vec_env("ACS-v0")

        model_path = f"{best_model_save_path}/best_model" if use_best_model else model_save_path
        model = PPO.load(model_path, env)
        model.set_logger(logger=logger)

    if perform_test:
        env = make_vec_env("ACS-v0")
        model.set_env(env)
        mean_rewards, std_rewards = test_agent(model, env, n_eval_episodes=test_n_eval_episodes)
        logger.info(f"Random Performance: Rewards: {mean_rewards} +/- {std_rewards:.2f}")

        env = Monitor(get_ideal_scenario_env())
        model.set_env(env)
        mean_rewards, std_rewards = test_agent(model, env, n_eval_episodes=test_n_eval_episodes)
        logger.info(f"Ideal Performance: Rewards: {mean_rewards} +/- {std_rewards:.2f}")

    if run_env:
        _perform_agent_runs(logger, model, 1)


def run_agent(model_path: str = "models/trained_model_v2", n_episodes: int = 1, use_ideal_scenario: bool = True):
    # Init logger
    logger: Logger = configure(None, ["stdout"])

    path = Path(model_path)
    if path.suffix == ".zip" and not path.exists():
        raise ValueError("The model_path must be a valid path to a .zip file")
    elif path.suffix != ".zip" and path.suffix != "":
        raise ValueError("The model_path must be a valid path to a .zip file "
                         "OR valid path without the .zip extension")

    model = PPO.load(path)
    model.set_logger(logger=logger)
    _perform_agent_runs(logger, model, n_episodes, use_ideal_scenario)


PERFORM_TRAINING = True
PERFORM_TESTING = True
RUN_ENV = False

if __name__ == "__main__":
    start_time = time.time()
    # perform_optuna_optimizing()
    train_agent(PERFORM_TRAINING, PERFORM_TESTING, RUN_ENV)
    print(f"Finished program. Execution time: {timedelta(seconds=(time.time() - start_time))}")
