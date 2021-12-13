import time
from datetime import datetime, timedelta

# noinspection PyUnresolvedReferences
from ac_carrier_scenario.ai.agent_training import run_agent
from ac_carrier_scenario.api.api import app

if __name__ == "__main__":
    start_time = time.time()
    # perform_optuna_optimizing()
    # run_agent(perform_training=True, perform_test=True, run_env=False)
    app.run()  # For dev
    print(f"[{datetime.now()}] Finished program. Execution time: {timedelta(seconds=(time.time() - start_time))}")
