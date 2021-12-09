import time
from datetime import timedelta

from agent_training import perform_optuna_optimizing

if __name__ == "__main__":
    start_time = time.time()
    perform_optuna_optimizing()
    print(f"Finished program. Execution time: {timedelta(seconds=(time.time() - start_time))}")
