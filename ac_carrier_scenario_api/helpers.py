from typing import Optional

from ac_carrier_scenario.scenarios import AircraftCarrierScenario

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
    scenario = AircraftCarrierScenario(missile_count: missile_count, jet_count: jet_count, pilot_count: pilot_count,
    target1_expected_damage: target1_expected_damage, target2_expected_damage: target2_expected_damage,
    target3_expected_damage: target3_expected_damage, target4_expected_damage: target4_expected_damage,
    target5_expected_damage: target5_expected_damage, target6_expected_damage: target6_expected_damage)
    return scenario