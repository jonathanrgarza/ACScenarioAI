"""
Contains the classes related to the aircraft carrier scenario gym environment.
"""
import random
from gym import Env
from gym.spaces import Discrete, Dict, MultiBinary, MultiDiscrete
import numpy as np
from aircraft_carrier_scenario import AircraftCarrierScenario


NUMBER_OF_TARGETS: int = 6
DEFENSE_ARRAY: list = [
    "target1Defense", "target2Defense", "target3Defense",
    "target4Defense", "target5Defense", "target6Defense"
]
SORTIE_ARRAY: list = [
    "target1Targets", "target2Targets", "target3Targets",
    "target4Targets", "target5Targets", "target6Targets",
]


def _get_init_state(scenario: AircraftCarrierScenario) -> dict:
    """
    Gets a specific initial state for AircraftCarrierScenarioEnv.

    :param scenario: The scenario parameters
    :type scenario: AircraftCarrierScenario
    :return: The new initial state for AircraftCarrierScenarioEnv.
    :rtype: dict
    """
    return {
        "missiles": scenario.missile_count,
        "expectedShipDamage":
            np.array([
                scenario.target1_expected_damage, scenario.target2_expected_damage,
                scenario.target3_expected_damage, scenario.target4_expected_damage,
                scenario.target5_expected_damage, scenario.target6_expected_damage
            ]),
        "currentShipDamage": np.array([0, 0, 0, 0, 0, 0]),
        "target1Defense": np.array([0, 0, 25, 0, 15, 0]),
        "target2Defense": np.array([0, 0, 0, 25, 0, 15]),
        "target3Defense": np.array([0, 0, 30, 0, 0, 0]),
        "target4Defense": np.array([0, 0, 0, 30, 0, 0]),
        "target5Defense": np.array([0, 0, 0, 0, 40, 0]),
        "target6Defense": np.array([0, 0, 0, 0, 0, 40]),
        "target1Targets": np.array([1, 0, 1, 0, 0, 0]),
        "target2Targets": np.array([0, 1, 0, 1, 0, 0]),
        "target3Targets": np.array([1, 0, 1, 0, 0, 0]),
        "target4Targets": np.array([0, 1, 0, 1, 0, 0]),
        "target5Targets": np.array([0, 0, 0, 0, 1, 1]),
        "target6Targets": np.array([0, 0, 0, 0, 1, 1]),
        "assets": np.array([scenario.jet_count, scenario.pilot_count])
    }


def _get_randomized_state() -> dict:
    """
    Gets a randomized initial state for AircraftCarrierScenarioEnv.

    :return: The new initial state for AircraftCarrierScenarioEn
    :rtype: dict
    """
    scenario = AircraftCarrierScenario(
        random.randint(1, 99), random.randint(1, 99), random.randint(1, 99),
        random.randint(0, 4), random.randint(0, 4), random.randint(0, 4),
        random.randint(0, 4), random.randint(0, 4), random.randint(0, 4))
    return _get_init_state(scenario)


class AircraftCarrierScenarioEnv(Env):
    """
    Environment representing an (random) aircraft carrier scenario.
    """
    def __init__(self) -> None:
        self.action_space = MultiDiscrete([6, 6, 2, 2])
        self.observation_space = Dict(
            {
                "missiles": Discrete(100),
                "expectedShipDamage": MultiDiscrete([100, 100, 100, 100, 100, 100]),
                "currentShipDamage": MultiDiscrete([100, 100, 100, 100, 100, 100]),
                "target1Defense": MultiDiscrete([100, 100, 100, 100, 100, 100]),
                "target2Defense": MultiDiscrete([100, 100, 100, 100, 100, 100]),
                "target3Defense": MultiDiscrete([100, 100, 100, 100, 100, 100]),
                "target4Defense": MultiDiscrete([100, 100, 100, 100, 100, 100]),
                "target5Defense": MultiDiscrete([100, 100, 100, 100, 100, 100]),
                "target6Defense": MultiDiscrete([100, 100, 100, 100, 100, 100]),
                "target1Targets": MultiBinary(6),
                "target2Targets": MultiBinary(6),
                "target3Targets": MultiBinary(6),
                "target4Targets": MultiBinary(6),
                "target5Targets": MultiBinary(6),
                "target6Targets": MultiBinary(6),
                "assets": MultiDiscrete([100, 100])
            })
        self.state = None
        self.reset()

    def render(self, mode="human") -> None:
        """
        Renders the environment. Not currently implemented.

        :param mode: The render mode.
        :type mode: str
        """
        super().render()  # Just throws an exception

    def _defend_ship(self, ship: int) -> bool:
        shot_down = False
        ship_defense_key = DEFENSE_ARRAY[ship]
        length = len(self.state[ship_defense_key])
        for index in range(0, length):
            if self.state[ship_defense_key][index] > 0:
                roll = random.randint(0, 100)
                if roll <= self.state[ship_defense_key][index]:
                    shot_down = True
        return shot_down

    def _can_attack(self, ship1: int, ship2: int) -> tuple[bool, int]:
        can_attack = False
        reward = -1000
        # Check if ship1 can attack ship2
        for sortie in SORTIE_ARRAY:
            if self.state[sortie][ship1] and self.state[sortie][ship2]:
                can_attack = True
                reward = 20
                break
        return can_attack, reward

    def _should_attack(self, ship: int) -> tuple[bool, int]:
        should_attack = True
        # Check if ship is already more damaged than expected/needed
        if self.state["currentShipDamage"][ship] >= self.state["expectedShipDamage"][ship]:
            # Check to see if target posses a threat
            if (self.state["currentShipDamage"][ship] == 0 and
                    self.state[DEFENSE_ARRAY[ship]][ship]):
                # Reward for damaging a ship that is a threat?
                reward = 0
            else:
                # Wasn't a threat and already at expected damage
                reward = -200
                should_attack = False
        else:
            # Targeted a ship worth targeting
            reward = 100
        return should_attack, reward

    def _shoot_ship(self, ship: int) -> int:
        reward = 0

        _, should_attack_reward = self._should_attack(ship)
        reward += should_attack_reward

        self.state["currentShipDamage"][ship] += 1
        self.state["missiles"] -= 1

        # Currently no reward for shooting a ship, all comes from _should_attack

        return reward

    def step(self, action: list[int]) -> tuple[dict, int, bool, dict]:
        """
        Performs a step in the environment.

        :param action: The action to take for this step.
        :return: The observation dictionary, the reward, if done, and a info dictionary.
        """
        info = {}
        reward: int = 0
        done: bool = False

        ship1_index = action[0]
        ship2_index = action[1]

        if action[2] == 1:
            reward += 10
        if action[3] == 1:
            reward += 10

        can_attack, can_attack_reward = self._can_attack(ship1_index, ship2_index)
        reward += can_attack_reward
        if can_attack:
            shot_down1 = self._defend_ship(ship1_index)
            shot_down2 = self._defend_ship(ship2_index)

            if ship1_index == ship2_index:
                if shot_down1:
                    self.state["assets"][0] -= 1
                    self.state["assets"][1] -= 1
            else:
                if shot_down1:
                    self.state["assets"][0] -= 1
                    self.state["assets"][1] -= 1
                if shot_down2:
                    self.state["assets"][0] -= 1
                    self.state["assets"][1] -= 1

            self._shoot_ship(ship1_index)

            if action[2] == 1:
                self._shoot_ship(ship1_index)

            self._shoot_ship(ship2_index)

            if action[3] == 1:
                self._shoot_ship(ship2_index)
        else:
            # Can't attack we are done
            return self.state, can_attack_reward, False, info

        # ############ Ideas ##############
        # Count up assets instead of down. Add a cap and remove end condition for assets
        # Change Reward/Loss system for expected damage
        #   Expected damage is now Expected hits. Remove Rolls
        # Add a calculated ratio reward for threats in the defending arrays
        # Add in standard deviation for rewards to encourage spreading out more hits when missiles are available
        #    Prio higher expected hits targets

        # Add penalty for going over an assets

        # Add penalty for going over missiles count
        if self.state["missiles"] < 0:
            reward -= 2000
        # Add penalty for going over jet count
        if self.state["assets"][0] < 0:
            reward -= 200
        # Add penalty for going over jet count
        if self.state["assets"][1] < 0:
            reward -= 200

        is_expected_damage_met = True
        for ship_index in range(0, NUMBER_OF_TARGETS):
            if (self.state["currentShipDamage"][ship_index] <
                    self.state["expectedShipDamage"][ship_index]):
                is_expected_damage_met = False
                break

        if is_expected_damage_met:
            # Reward for meeting the expected damage
            reward += 100
            # Reward for each missile remaining
            reward += max(0, self.state["missiles"]) * 10
            # Reward for each jet remaining
            reward += max(0, self.state["assets"][0]) * 5
            # Reward for each pilot remaining
            reward += max(0, self.state["assets"][1]) * 5

        if self.state["missiles"] <= 0:
            done = True

        if self.state["assets"][0] <= 0 or self.state["assets"][1] <= 0:
            done = True

        # Return step information
        return self.state, reward, done, info

    def reset(self) -> None:
        """
        Resets the environment to an initial state.
        """
        self.state = _get_randomized_state()


class SpecificAircraftCarrierScenarioEnv(AircraftCarrierScenarioEnv):
    """
    Environment representing a specific (non-random) aircraft carrier scenario.
    """
    def __init__(self, scenario: AircraftCarrierScenario) -> None:
        super().__init__()

        if scenario is not AircraftCarrierScenario:
            raise ValueError("scenario is not of the expected type 'AircraftCarrierScenario'")
        self._scenario = scenario
        self.reset()

    @property
    def scenario(self) -> AircraftCarrierScenario:
        """
        The scenario property.

        :return: The scenario being used by this environment.
        :rtype: AircraftCarrierScenario
        """
        return self._scenario

    def reset(self) -> None:
        """
        Resets the environment to the initial state.
        """
        self.state = _get_init_state(self._scenario)
