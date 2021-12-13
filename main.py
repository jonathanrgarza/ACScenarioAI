import time
from datetime import timedelta

from ac_carrier_scenario.ai.agent_training import run_agent

if __name__ == "__main__":
    start_time = time.time()
    # perform_optuna_optimizing()
    run_agent(perform_training=True, perform_test=True, run_env=False)
    print(f"Finished program. Execution time: {timedelta(seconds=(time.time() - start_time))}")
