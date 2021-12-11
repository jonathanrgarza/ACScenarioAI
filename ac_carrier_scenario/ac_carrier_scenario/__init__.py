from gym.envs.registration import register

# Register env with gym
register(
    id='ACS-v0',
    entry_point='ac_carrier_scenario.environment:AircraftCarrierScenarioEnv',
    max_episode_steps=250
)
