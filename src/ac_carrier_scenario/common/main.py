import argparse
import sys
import time
from datetime import datetime, timedelta
# noinspection PyUnresolvedReferences
from typing import Optional, List, Union, Any

from ac_carrier_scenario.ai.agent_training import run_agent, perform_optuna_optimizing, train_agent
# noinspection PyUnresolvedReferences
from ac_carrier_scenario.api.api import app


def _add_agent_options(agent: argparse.ArgumentParser) -> None:
    # Add AI command options
    subparser = agent.add_subparsers(dest="sub_command")

    # Add optimize options
    optimize = subparser.add_parser("optimize")
    optimize.add_argument("--study_name", default="agent_study_1",
                          help="The name of the study. Determines the part of file names")
    optimize.add_argument("--n_trials", type=int, default=100,
                          help="The number of trials to run")
    optimize.add_argument("--n_envs", type=int, default=3,
                          help="The number of parallel environments used during training")
    optimize.add_argument("--total_timesteps", type=int, default=25000,
                          help="The total number of timesteps for the training")
    optimize.add_argument("--eval_freq", type=int, default=10000,
                          help="The number of time steps between evaluations")
    optimize.add_argument("--n_eval_episodes", type=int, default=5,
                          help="The number of episodes used for evaluation")
    optimize.add_argument("--tb_log_path", default=None,
                          help="The path to the folder to put tensorboard logs during optimization")
    optimize.add_argument("--verbose_learning", type=int, default=0,
                          help="The learning verbose output level. 0 = None, 1 = info and 2 = debug")
    optimize.add_argument("--verbose_eval", type=int, default=1,
                          help="The evaluation verbose output level. 0 = None, 1 = info and 2 = debug")

    # Add train options
    train = subparser.add_parser("train")
    train.add_argument("--model_save_path", default="models/trained_model_v2",
                       help="The path to the model file")
    train.add_argument("--best_model_save_path", default="models",
                       help="The path to the folder to save the best model found during training")
    train.add_argument("--tb_log_path", default=None,
                       help="The path to the folder to put tensorboard logs during training")
    train.add_argument("--n_envs", type=int, default=3,
                       help="The number of parallel environments used during training")
    train.add_argument("--total_timesteps", type=int, default=25000,
                       help="The total number of timesteps for the training")
    train.add_argument("--n_eval_episodes", type=int, default=5,
                       help="The number of episodes used for evaluation")
    train.add_argument("--eval_freq", type=int, default=10000,
                       help="The number of time steps between evaluations")
    train.add_argument("--skip_training", action="store_true",
                       help="Should the training of the model be skipped")
    train.add_argument("--perform_test", action="store_true",
                       help="Should an test/evaluation of the model be run after training")
    train.add_argument("--test_n_eval_episodes", type=int, default=10,
                       help="The number of episodes used for evaluation during testing")
    train.add_argument("--perform_run", action="store_true",
                       help="Should a run of the model be run after training")
    train.add_argument("--use_trained_model", action="store_false",
                       help="Use the saved trained model instead of the best model for test and/or run")
    train.add_argument("--verbose_learning", type=int, default=0,
                       help="The learning verbose output level. 0 = None, 1 = info and 2 = debug")
    train.add_argument("--verbose_eval", type=int, default=1,
                       help="The evaluation verbose output level. 0 = None, 1 = info and 2 = debug")

    # Add enjoy options
    enjoy = subparser.add_parser("enjoy")
    enjoy.add_argument("--model_path", default="models/trained_model_v2",
                       help="The path to the model file")
    enjoy.add_argument("--n_episodes", type=int, default=1,
                       help="The number of episodes/runs to perform")
    enjoy.add_argument("--use_random_env", action="store_false",
                       help="Use a random environment for run(s). Otherwise will use the ideal scenario")


def _add_api_options(api: argparse.ArgumentParser) -> None:
    api.add_argument("--port", type=int, help="The port to use for the API server")


def _add_cli_parser_options(ap: argparse.ArgumentParser) -> None:
    if not isinstance(ap, argparse.ArgumentParser):
        raise ValueError("ap is not of type 'ArgumentParser'")

    subparser = ap.add_subparsers(dest="command")

    # Add AI command options
    _add_agent_options(subparser.add_parser("agent"))

    # Add API command options
    _add_api_options(subparser.add_parser("api"))


def run_ai_agent(arguments: argparse.Namespace):
    if arguments.sub_command == "optimize":
        perform_optuna_optimizing(study_name=arguments.study_name, n_trials=arguments.n_trials,
                                  n_envs=arguments.n_envs, total_timesteps=arguments.total_timesteps,
                                  n_eval_episodes=arguments.n_eval_episodes, eval_freq=arguments.eval_freq,
                                  learning_verbose_level=arguments.verbose_learning,
                                  eval_verbose_level=arguments.verbose_eval, tb_log_path=arguments.tb_log_path)
    elif arguments.sub_command == "train":
        perform_training = not arguments.skip_training
        train_agent(perform_training=perform_training, perform_test=arguments.perform_test,
                    run_env=arguments.perform_run, use_best_model=arguments.use_trained_model,
                    model_save_path=arguments.model_save_path, best_model_save_path=arguments.best_model_save_path,
                    tb_log_path=arguments.tb_log_path, n_envs=arguments.n_envs,
                    total_timesteps=arguments.total_timesteps, n_eval_episodes=arguments.n_eval_episodes,
                    eval_freq=arguments.eval_freq, learning_verbose_level=arguments.verbose_learning,
                    eval_verbose_level=arguments.verbose_eval, test_n_eval_episodes=arguments.test_n_eval_episodes)
    elif arguments.sub_command == "enjoy":
        run_agent(model_path=arguments.model_path, n_episodes=arguments.n_episodes,
                  use_ideal_scenario=arguments.use_random_env)
    else:
        raise ValueError(f"Invalid sub command for agent: {arguments.sub_command}")


def main(cli_args: List[str] = None) -> int:
    """
    Main entry point for ac_carrier_scenario
    :param cli_args: The command line arguments. Defaults to None (They will be gotten from sys.argv then).
    :return: The exit code. 0 for normal exit. Non-zero for exiting with an error.
    """
    # Get the command line arguments if none where provided (by another function calling this method)
    if cli_args is None:
        cli_args = sys.argv[1:]

    start_time = time.time()
    try:
        # Construct the argument parser
        ap = argparse.ArgumentParser()
        _add_cli_parser_options(ap)
        arguments = ap.parse_args(cli_args)

        if arguments.command == "agent":
            run_ai_agent(arguments)
        elif arguments.command == "api":
            app.run(port=arguments.port)  # For dev

    except KeyboardInterrupt:
        print('Aborted manually.', file=sys.stderr)
        print(f"[{datetime.now()}] Finished program. Execution time: {timedelta(seconds=(time.time() - start_time))}")
        return 1
    print(f"[{datetime.now()}] Finished program. Execution time: {timedelta(seconds=(time.time() - start_time))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
