"""
Contains the classes related to the aircraft carrier scenario.
"""
from typing import Optional


class AircraftCarrierScenario:
    """
    Represents an aircraft carrier scenario's parameters.
    """
    def __init__(self, missile_count: Optional[int] = None,
                 jet_count: Optional[int] = None, pilot_count: Optional[int] = None,
                 target1_expected_damage: Optional[int] = None,
                 target2_expected_damage: Optional[int] = None,
                 target3_expected_damage: Optional[int] = None,
                 target4_expected_damage: Optional[int] = None,
                 target5_expected_damage: Optional[int] = None,
                 target6_expected_damage: Optional[int] = None) -> None:
        """
        Initializes an aircraft carrier scenario.

        :param missile_count: The number of missiles. Defaults to None.
        :param jet_count: The number of jets. Defaults to None.
        :param pilot_count: The number of pilots. Default to None.
        :param target1_expected_damage: The expected damage for target 1. Defaults to None.
        :param target2_expected_damage: The expected damage for target 2. Defaults to None.
        :param target3_expected_damage: The expected damage for target 3. Defaults to None.
        :param target4_expected_damage: The expected damage for target 4. Defaults to None.
        :param target5_expected_damage: The expected damage for target 5. Defaults to None.
        :param target6_expected_damage: The expected damage for target 6. Defaults to None.
        """
        self.missile_count = missile_count or 0
        self.jet_count = jet_count or 0
        self.pilot_count = pilot_count or 0
        self.target1_expected_damage = target1_expected_damage or 0
        self.target2_expected_damage = target2_expected_damage or 0
        self.target3_expected_damage = target3_expected_damage or 0
        self.target4_expected_damage = target4_expected_damage or 0
        self.target5_expected_damage = target5_expected_damage or 0
        self.target6_expected_damage = target6_expected_damage or 0

    @property
    def missile_count(self) -> int:
        """
        The missile count property.

        :return: The number of missiles for the scenario.
        """
        return self._missile_count

    @missile_count.setter
    def missile_count(self, value: int) -> None:
        if value < 0:
            raise ValueError("missile_count value can not be below zero")
        self._missile_count = value

    @property
    def jet_count(self) -> int:
        """
        The jet count property.

        :return: The number of jets for the scenario.
        """
        return self._jet_count

    @jet_count.setter
    def jet_count(self, value: int) -> None:
        if value < 0:
            raise ValueError("jet_count value can not be below zero")
        self._jet_count = value

    @property
    def pilot_count(self) -> int:
        """
        The pilot count property.

        :return: The number of pilots for the scenario.
        """
        return self._pilot_count

    @pilot_count.setter
    def pilot_count(self, value: int) -> None:
        if value < 0:
            raise ValueError("pilot_count value can not be below zero")
        self._pilot_count = value

    @property
    def target1_expected_damage(self) -> int:
        """
        The expected damage for target 1 property.

        :return: The expected damage for target 1 in the scenario.
        """
        return self._target1_expected_damage

    @target1_expected_damage.setter
    def target1_expected_damage(self, value: int) -> None:
        if value < 0:
            raise ValueError("target1_expected_damage value can not be below zero")
        self._target1_expected_damage = value

    @property
    def target2_expected_damage(self) -> int:
        """
        The expected damage for target 2 property.

        :return: The expected damage for target 2 in the scenario.
        """
        return self._target2_expected_damage

    @target2_expected_damage.setter
    def target2_expected_damage(self, value: int) -> None:
        if value < 0:
            raise ValueError("target2_expected_damage value can not be below zero")
        self._target2_expected_damage = value

    @property
    def target3_expected_damage(self) -> int:
        """
        The expected damage for target 3 property.

        :return: The expected damage for target 3 in the scenario.
        """
        return self._target3_expected_damage

    @target3_expected_damage.setter
    def target3_expected_damage(self, value: int) -> None:
        if value < 0:
            raise ValueError("target3_expected_damage value can not be below zero")
        self._target3_expected_damage = value

    @property
    def target4_expected_damage(self) -> int:
        """
        The expected damage for target 4 property.

        :return: The expected damage for target 4 in the scenario.
        """
        return self._target4_expected_damage

    @target4_expected_damage.setter
    def target4_expected_damage(self, value: int) -> None:
        if value < 0:
            raise ValueError("target4_expected_damage value can not be below zero")
        self._target4_expected_damage = value

    @property
    def target5_expected_damage(self) -> int:
        """
        The expected damage for target 5 property.

        :return: The expected damage for target 5 in the scenario.
        """
        return self._target5_expected_damage

    @target5_expected_damage.setter
    def target5_expected_damage(self, value: int) -> None:
        if value < 0:
            raise ValueError("target5_expected_damage value can not be below zero")
        self._target5_expected_damage = value

    @property
    def target6_expected_damage(self) -> int:
        """
        The expected damage for target 6 property.
        
        :return: The expected damage for target 6 in the scenario.
        """
        return self._target6_expected_damage

    @target6_expected_damage.setter
    def target6_expected_damage(self, value: int) -> None:
        if value < 0:
            raise ValueError("target6_expected_damage value can not be below zero")
        self._target6_expected_damage = value
