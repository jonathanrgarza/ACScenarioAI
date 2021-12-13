from typing import Optional

import numpy as np
from flask import Response, jsonify
from gym import Env
from gym.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm

from ac_carrier_scenario.common.environment import SpecificAircraftCarrierScenarioEnv
from ac_carrier_scenario.common.scenarios import AircraftCarrierScenario


MODEL_PATH = "models/best_model"


def get_scenario_from_json(json: dict) -> Optional[AircraftCarrierScenario]:
    if not isinstance(json, dict):
        return None

    if "missile_count" not in json:
        return None
    missile_count: int = int(json["missile_count"])
    if "jet_count" not in json:
        return None
    jet_count: int = int(json["jet_count"])
    if "pilot_count" not in json:
        return None
    pilot_count: int = int(json["pilot_count"])
    if "target1_expected_damage" not in json:
        return None
    target1_expected_damage: int = int(json["target1_expected_damage"])
    if "target2_expected_damage" not in json:
        return None
    target2_expected_damage: int = int(json["target2_expected_damage"])
    if "target3_expected_damage" not in json:
        return None
    target3_expected_damage: int = int(json["target3_expected_damage"])
    if "target4_expected_damage" not in json:
        return None
    target4_expected_damage: int = int(json["target4_expected_damage"])
    if "target5_expected_damage" not in json:
        return None
    target5_expected_damage: int = int(json["target5_expected_damage"])
    if "target6_expected_damage" not in json:
        return None
    target6_expected_damage: int = int(json["target6_expected_damage"])

    # Create AircraftCarrierScenario from json values
    scenario = AircraftCarrierScenario(missile_count=missile_count, jet_count=jet_count, pilot_count=pilot_count,
                                       target1_expected_damage=target1_expected_damage,
                                       target2_expected_damage=target2_expected_damage,
                                       target3_expected_damage=target3_expected_damage,
                                       target4_expected_damage=target4_expected_damage,
                                       target5_expected_damage=target5_expected_damage,
                                       target6_expected_damage=target6_expected_damage)
    return scenario


def perform_analysis(scenario: AircraftCarrierScenario) -> Optional[dict]:
    # Create the specific env from the given scenario
    # ENSURE TO WRAP ENV in TimeLimit wrapper
    env: Env = TimeLimit(env=SpecificAircraftCarrierScenarioEnv(scenario), max_episode_steps=250)

    # Load the agent/model
    try:
        model = PPO.load(MODEL_PATH, env)
    except FileNotFoundError:
        return None

    monitored_env: Monitor = Monitor(env)
    n_eval_episodes = 10
    mean_solves, mean_episode_length, mean_rewards, std_rewards = _get_analysis(
        model, monitored_env, n_eval_episodes=n_eval_episodes)

    return {
        "mean_solves": mean_solves, "mean_episode_length": mean_episode_length,
        "mean_rewards": mean_rewards, "std_rewards": std_rewards,
        "total_episodes": n_eval_episodes
    }


def _get_analysis(model: OnPolicyAlgorithm, monitored_env: Monitor,
                  n_eval_episodes: int = 10) -> tuple[float, float, float, float]:
    solved_env_count: int = 0
    rewards: list[float] = []
    step_lengths: list[int] = []

    if n_eval_episodes < 1:
        raise ValueError("n_eval_episodes can not be less than 1")

    for _ in range(n_eval_episodes):
        state = monitored_env.reset()
        done = False

        reward_score = 0
        steps = 0
        # Run the episode til completion
        for _ in range(256):
            action, _states = model.predict(state)
            new_state, reward, done, info = monitored_env.step(action)
            state = new_state

            reward_score += reward
            steps += 1

            if done:
                break

        # Add results from episode to overall results
        env: SpecificAircraftCarrierScenarioEnv = monitored_env.unwrapped
        if done and env.is_expected_damage_met:
            solved_env_count += 1

        rewards.append(reward_score)
        step_lengths.append(steps)

    # Determine statistic results of overall results

    mean_solved_envs: float = solved_env_count / n_eval_episodes
    # noinspection PyTypeChecker
    mean_episode_length: float = np.mean(step_lengths)
    # noinspection PyTypeChecker
    mean_rewards: float = np.mean(rewards)
    # noinspection PyTypeChecker
    std_rewards: float = np.std(rewards)

    return mean_solved_envs, mean_episode_length, mean_rewards, std_rewards


def get_performance_stats(scenario: AircraftCarrierScenario) -> Optional[dict]:
    # Create the specific env from the given scenario
    # ENSURE TO WRAP ENV in TimeLimit wrapper
    env: Env = TimeLimit(env=SpecificAircraftCarrierScenarioEnv(scenario), max_episode_steps=250)

    # Load the agent/model
    try:
        model = PPO.load(MODEL_PATH, env)
    except FileNotFoundError:
        return None

    monitored_env: Monitor = Monitor(env)
    n_eval_episodes = 10
    mean_rewards, std_rewards = evaluate_policy(
        model, monitored_env, n_eval_episodes=n_eval_episodes)

    return {
        "mean_rewards": mean_rewards, "std_rewards": std_rewards, "total_episodes": n_eval_episodes
    }


def get_flask_response(is_valid_request: bool, is_valid_scenario: bool, results: Optional[dict]) -> Response:
    if not is_valid_request:
        response = jsonify("{error:\"No JSON data was submitted or mimetype was not 'application/json'\"}")
        response.status_code = 400  # Bad Request
    elif not is_valid_scenario:
        response = jsonify("{error:\"Not a valid scenario\"}")
        response.status_code = 400  # Bad Request
    else:
        if results is None:
            response = jsonify("{error:\"API encountered an unexpected error\"}")
            response.status_code = 500  # Internal Server Error
        else:
            response = jsonify(results)
            response.status_code = 200
    return response
