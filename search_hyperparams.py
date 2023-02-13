#!/usr/bin/env python3

"""Peform hyperparemeters search"""

import os
import itertools
import json
from tqdm import tqdm
from subprocess import check_call
from argparse import Namespace
import time
from typing import Optional

import config
from train import get_train_parser


def create_job_dir(root_dir: str, job_name: str, params=Optional[dict]) -> str:

    # Create a new folder in parent_dir with unique_name "job_name"
    job_dir = os.path.join(root_dir, job_name)
    if not os.path.exists(job_dir):
        os.makedirs(job_dir)

    # Write parameters in json file
    if params is not None:
        json_path = os.path.join(job_dir, "params.json")
        with open(json_path, "w") as f:
            json.dump(params, f)

    return job_dir


def get_hyperparam_parser():
    """This function extends the training parser with the parameters
    required for tuning the hyperparameters.
    """

    # Get the parser for the train script
    parser = get_train_parser()

    # Extend the argument parser
    """
    parser.add_argument(
        "--experiment_dir",
        default="experiments/learning_rate",
        help="Directory for reporting the model experiments",
    )
    """
    parser.add_argument(
        "--parameter_space",
        type=str,
        default="hyperparameter_space/example.json",
        help="Path to json file describing the parameter space",
    )
    parser.add_argument(
        "--rerun",
        default=False,
        action="store_true",
        help="Force rerunning of a config even if the job dir already exists",
    )
    return parser


def launch_training_job(job_dir: str, args: Namespace) -> None:
    """
    Launch training of a model given the directory path containing configuration file
    and a namespace containing the training parameters.
    """

    # The base command
    cmd = f"{config.PYTHON} train.py "

    # Get the default parameters from the training script
    dummy_parser = get_train_parser()
    default_parameters = dummy_parser.parse_args([]).__dict__.keys()

    # Add all other remaining training parameters
    for i, param in enumerate(default_parameters, start=1):

        if isinstance(
            dummy_parser.__dict__["_actions"][i], dummy_parser.__dict__["_registries"]["action"]["store_true"]
        ):
            # Parse ActionStoreTrue option
            if args.__dict__[param]:  # if set to true - add flag
                cmd += f" --{param} "
        elif isinstance(
            dummy_parser.__dict__["_actions"][i],
            dummy_parser.__dict__["_registries"]["action"]["store_false"],
        ):
            # Parse ActionStoreFalse option
            if not args.__dict__[param]:  # if set to false - add flag
                cmd += f" --{param} "
        elif param == "experiment_dir":
            cmd += f" --{param} {job_dir}"
        else:
            if args.__dict__[param] is not None:  # Cannot parse none values
                # Include option and value
                cmd += f" --{param} {args.__dict__[param]} "

    print(cmd)
    check_call(cmd, shell=True, stdout=None)


def main():

    parser = get_hyperparam_parser()
    args = parser.parse_args()

    # Define the hyperparameter space
    with open(args.parameter_space, "r") as f:
        hp_space = json.load(f)

    # Cartesian product of the parameter space
    hp_parameters = [dict(zip(hp_space, v)) for v in itertools.product(*hp_space.values())]

    # Iterate over each param config
    no_skipped_runs = 0
    for param_config in tqdm(hp_parameters):

        # Make config description - limit to only variable parameters
        job_name = "_".join(
            [
                p + "_" + str(v).replace(" ", "_")  # Cater for multiple axiom orders
                for p, v in sorted(param_config.items())
                if len(hp_space[p]) > 1
            ]
        )

        print(f"\n### Processing job: {job_name}")

        # If we are not forcing reruns and the jobdir already exists, we skip this configuration
        if not args.rerun and os.path.exists(os.path.join(args.experiment_dir, job_name)):
            no_skipped_runs += 1
            print(f"Skipping rerun of: {job_name}")
            continue

        # Create job dir
        job_dir = create_job_dir(args.experiment_dir, job_name, params=param_config)

        # Update the model directory
        args.model_dir = job_dir

        # Launch job
        launch_training_job(job_dir, args)
        time.sleep(2)  # Give the machine two seconds to recover

    # Report the number of skipped runs if any
    if no_skipped_runs > 0:
        print(f"Skipped a total of {no_skipped_runs} job runs")
    print("Finished")


if __name__ == "__main__":

    main()
