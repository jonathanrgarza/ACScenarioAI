from typing import Optional

import numpy as np
import gym
from gym.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from ac_carrier_scenario.common.scenarios import AircraftCarrierScenario
from ac_carrier_scenario.common.environment import SpecificAircraftCarrierScenarioEnv


def get_scenario_from_json(json: dict) -> Optional[AircraftCarrierScenario]:
    if json is not dict:
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
        model = PPO.load("models/trained_model_more", env)
    except FileNotFoundError:
        return None

    monitored_env: Monitor = Monitor(env)
    mean_solves, mean_rewards, std_rewards = _get_analysis(model, monitored_env, n_eval_episodes=10)

    return {"mean_solves": mean_solves, "mean_rewards": mean_rewards, "std_rewards": std_rewards}


def _get_analysis(model: OnPolicyAlgorithm, monitored_env: Monitor,
                  n_eval_episodes: int = 10) -> tuple[float, float, float]:
    solved_env_count: int = 0
    rewards: list[float] = []

    if n_eval_episodes < 1:
        raise ValueError("n_eval_episodes can not be less than 1")

    for _ in range(n_eval_episodes):
        state = monitored_env.reset()
        done = False

        reward_score = 0
        steps = 0
        for _ in range(256):
            action, _states = model.predict(state)
            new_state, reward, done, info = monitored_env.step(action)
            state = new_state

            reward_score += reward
            steps += 1

            time.sleep(0.3)

            if done:
                break

        env: SpecificAircraftCarrierScenarioEnv = monitored_env.unwrapped
        if done and env.is_expected_damage_met:
            solved_env_count += 1

        rewards.append(reward_score)

        # noinspection PyTypeChecker
        mean_rewards: float = np.mean(rewards)
        # noinspection PyTypeChecker
        std_rewards: float = np.std(rewards)
        mean_solved_envs: float = solved_env_count / n_eval_episodes

        return mean_solved_envs, mean_rewards, std_rewards
