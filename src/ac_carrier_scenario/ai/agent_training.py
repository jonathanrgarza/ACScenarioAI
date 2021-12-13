"""
Module for training the AI agent.
"""
import logging
import os
import sys
import time
from datetime import timedelta
from typing import Optional, Union

import gym
import optuna
from gym import Env
from optuna.visualization import plot_optimization_history, plot_param_importances
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.callbacks import StopTrainingOnRewardThreshold, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import Logger, configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecEnv

import ac_carrier_scenario.util.stay_awake as stay_awake
from ac_carrier_scenario.ai.trial_eval_callback import TrialEvalCallback

# Import the custom env
import ac_carrier_scenario.common  # This is required to register the gym environment


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


def agent_objective(trial: optuna.Trial) -> int:
    """
    Function to use with optuna to tune hyperparameters for ACS environment.

    :param trial: The optuna trial
    :return: The mean reward for the trial
    :rtype: int
    """
    # Create the gym model
    gym_env: Env = gym.make("ACS-v0")
    eval_env = Monitor(gym_env)

    # Determine the hyperparameters
    # algorithm = trial.suggest_categorical("algorithm", ["PPO", "A2C"])
    algorithm = "PPO"
    # policy = "MlpPolicy"
    policy = "MultiInputPolicy"

    # Get trial's hyperparameters that are common to all algorithms
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1)
    n_steps = trial.suggest_categorical("n_steps", [8, 16, 32, 64, 128, 256, 512, 1024, 2048])
    gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    ent_coef = trial.suggest_loguniform("ent_coef", 0.00000001, 0.1)
    vf_coef = trial.suggest_uniform("vf_coef", 0, 1)
    gae_lambda = trial.suggest_categorical("gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])
    max_grad_norm = trial.suggest_categorical("max_grad_norm", [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5])

    # Launch tensorboard with command: tensorboard --logdir=models/optuna/logging

    if algorithm == "PPO":
        # Get trial's hyperparameters that are for PPO algorithm only
        clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3, 0.4])
        n_epochs = trial.suggest_categorical("n_epochs", [1, 5, 10, 20])

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

        model = PPO(policy, eval_env, learning_rate=learning_rate, gamma=gamma, n_steps=n_steps, n_epochs=n_epochs,
                    batch_size=batch_size, ent_coef=ent_coef, clip_range=clip_range,
                    gae_lambda=gae_lambda, max_grad_norm=max_grad_norm, vf_coef=vf_coef,
                    tensorboard_log="models/optuna/logging", verbose=0)
    elif algorithm == "A2C":
        # Get trial's hyperparameters that are for A2C algorithm only
        normalize_advantage = trial.suggest_categorical("normalize_advantage", [False, True])
        # Toggle PyTorch RMS Prop (different from TF one, cf doc)
        use_rms_prop = trial.suggest_categorical("use_rms_prop", [False, True])

        model = A2C(policy, eval_env, learning_rate=learning_rate, n_steps=n_steps, gamma=gamma, gae_lambda=gae_lambda,
                    ent_coef=ent_coef, vf_coef=vf_coef, max_grad_norm=max_grad_norm, use_rms_prop=use_rms_prop,
                    normalize_advantage=normalize_advantage,
                    tensorboard_log="models/optuna/logging", verbose=0)
    else:
        raise ValueError(f"Invalid algorithm selected: {algorithm}")

    eval_callback = TrialEvalCallback(eval_env, trial, verbose=0)

    try:
        # No keep awake needed here as this is called by optuna which has been kept awake
        print(f"Starting a trial '{trial.number}' using the '{algorithm}' algorithm")
        model.learn(25000 * 5, callback=eval_callback)

        model.env.close()
        eval_env.close()
    except (AssertionError, ValueError) as e:
        # Sometimes, random hyperparameters can generate NaN
        model.env.close()
        eval_env.close()
        # Prune hyperparameters that generate NaNs
        print(e)
        print("============")
        print("Sampled hyperparameters:")
        print(trial.params)
        raise optuna.exceptions.TrialPruned()

    is_pruned = eval_callback.is_pruned
    reward = eval_callback.last_mean_reward

    del model.env, eval_callback
    del model

    if is_pruned:
        raise optuna.exceptions.TrialPruned()

    return reward


def perform_optuna_optimizing(n_trials: int = 100):
    print("Initializing an optuna hyperparameter optimization study run")

    # Create dir if needed
    try:
        os.makedirs("models/optuna", exist_ok=True)
    except OSError:
        print("Could not create folder 'models/optuna'. Create this folder first and try again.")
        exit(1)

    # Add stream handler of stdout to show the messages
    logger = optuna.logging.get_logger("optuna")
    log_handler = logging.StreamHandler(sys.stdout)
    log_handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s"))
    logger.addHandler(log_handler)
    study_name = "agent_study_2"  # Unique identifier of the study
    storage_name = f"sqlite:///models/optuna/{study_name}.db"
    study = optuna.create_study(study_name=study_name, storage=storage_name, direction="maximize", load_if_exists=True)

    completed = False

    try:
        with stay_awake.keep_awake():
            logger.info("Starting an optuna hyperparameter optimization run")
            study.optimize(agent_objective, n_trials=n_trials)
            completed = True
    except KeyboardInterrupt:
        pass

    logger.info("Number of finished trials: ", len(study.trials))

    trial = study.best_trial
    logger.info(f"Best trial: {trial.number}")

    logger.info("Value: ", trial.value)

    logger.info("Params: ")
    for key, value in trial.params.items():
        logger.info(f"    {key}: {value}")

    if completed:
        completed_str = "Completed"
    else:
        completed_str = "Uncompleted"

    report_name = (
        f"report_agent_{n_trials}-trials-{25000}"
        f"-TPE-None_{int(time.time())}-status-{completed_str}"
    )

    log_path = os.path.join("models", "optuna", report_name)

    logger.info(f"Writing report to {log_path}")

    # Write report
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    study.trials_dataframe().to_csv(f"{log_path}.csv")

    # Plot optimization result
    try:
        fig1 = plot_optimization_history(study)
        fig2 = plot_param_importances(study)

        fig1.show()
        fig2.show()
    except (ValueError, ImportError, RuntimeError) as e:
        logger.warning(f"Could not plot study: {e}")


def test_agent(model, env: Env):
    monitored_env = Monitor(env)
    model.set_env(monitored_env)
    return evaluate_policy(model, monitored_env, n_eval_episodes=10)


def get_new_ppo_agent(env: Union[Env, VecEnv, str],
                      tensorboard_log: Optional[str] = None, verbose: bool = False) -> PPO:
    """
    Gets a new PPO agent for the given environment. This is meant only for the AC Carrier Scenario.

    :param env: The gym environment.
    :param tensorboard_log: The tensorboard log location. Default is None.
    :param verbose: The verbose setting. Default is False.
    :return: The PPO agent.
    """
    if env is None:
        raise ValueError("The env is not set")

    policy = "MultiInputPolicy"  # This policy is required for Dict type of environments

    # Set the best hyperparams found; value reached for these parameters: 1744
    learning_rate = 0.005414873093748744
    n_steps = 256
    gamma = 0.9
    ent_coef = 1.992566725293565e-05
    clip_range = 0.4
    n_epochs = 20
    gae_lambda = 0.98
    max_grad_norm = 0.9
    vf_coef = 0.3900021887374706
    batch_size = 128

    verbose_int = 0
    if verbose:
        verbose_int = 1

    model = PPO(policy, env, learning_rate=learning_rate, gamma=gamma, n_steps=n_steps, n_epochs=n_epochs,
                batch_size=batch_size, ent_coef=ent_coef, clip_range=clip_range,
                gae_lambda=gae_lambda, max_grad_norm=max_grad_norm, vf_coef=vf_coef,
                tensorboard_log=tensorboard_log, verbose=verbose_int)
    return model


def perform_agent_training(logger: Logger):
    # Parallel Environments
    env = make_vec_env("ACS-v0", n_envs=4)

    try:
        model = PPO.load("models/best_model", env)
    except FileNotFoundError:
        model = None
        logger.log("No existing model found at models/best_model.zip")

    if model is None:
        logger.log("No existing model. Creating a new model to learn with")
        model = get_new_ppo_agent(env, "models/logging", True)
    else:
        logger.log("Existing model found. Will continue its learning")

    model.set_logger(logger)

    # Set callbacks
    # Separate evaluation environment
    eval_env = Monitor(gym.make("ACS-v0"))
    # callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=10, verbose=1)
    callback_on_best = None
    eval_callback = EvalCallback(eval_env=eval_env, callback_on_new_best=callback_on_best,
                                 best_model_save_path="models", verbose=1)
    with stay_awake.keep_awake():
        model.learn(total_timesteps=25000 * 200, callback=eval_callback)

    model.save("models/trained_model")
    env.close()
    logger.log("Training complete")

    return model


def run_agent(perform_training: bool, perform_test: bool, run_env: bool):
    print("Running stable_baselines3 PPO agent learning")

    # Init logger
    logger = configure(None, ["stdout"])

    if not perform_training and not run_env:
        logger.log("No action to perform, please update script/program and run again")
        return

    if perform_training:
        model = perform_agent_training(logger)
    else:
        logger.log("Training option disabled. Loading model from file")
        env = gym.make("ACS-v0")
        model = PPO.load("models/best_model", env)

    if perform_test:
        env = gym.make("ACS-v0")
        model.set_env(env)
        mean_rewards, std_rewards = test_agent(model, env)
        logger.log(f"Performance: Rewards: {mean_rewards} +/- {std_rewards:.2f}")

    if run_env:
        env = gym.make("ACS-v0")
        model.set_env(env)
        state = env.reset()
        done = False

        reward_score = 0
        steps = 0
        for _ in range(1000):
            action, _states = model.predict(state)
            new_state, reward, done, info = env.step(action)
            state = new_state

            reward_score += reward
            steps += 1

            # clear_console()
            # env.render()
            logger.log(f"Step#: {steps}, Reward: {reward}")
            time.sleep(0.3)

            if done:
                logger.log(f"Reward Sum/Score: {reward_score}, Steps: {steps}")
                break

        if not done:
            logger.log("Agent could not complete the environment")

        env.close()
        logger.log("Run complete")


PERFORM_TRAINING = True
PERFORM_TESTING = True
RENDER_ENV = False

if __name__ == "__main__":
    start_time = time.time()
    # perform_optuna_optimizing()
    run_agent(PERFORM_TRAINING, PERFORM_TESTING, RENDER_ENV)
    print(f"Finished program. Execution time: {timedelta(seconds=(time.time() - start_time))}")