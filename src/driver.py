from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from pathlib import Path

from apu import config, setup_logger
import apu.utils

from learner import execute_test


def parse_args() -> Namespace:
    r""" Parse, checks, and refactors the input arguments"""
    args = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)  # noqa
    # noinspection PyTypeChecker
    args.add_argument("config_file", help="Path to the configuration file", type=Path)
    args.add_argument("-d", help="Debug mode -- Disable non-determinism", action="store_true")
    args.add_argument("--bias_priors", help="Bias the prior probabilities",
                      action="store_true")
    args = args.parse_args()

    if not args.config_file.exists() or not args.config_file.is_file():
        raise ValueError(f"Unknown configuration file \"{args.config_file}\"")

    config.parse(args.config_file)
    args.d |= config.DATASET.is_synthetic()

    seed = 42 if args.d else None
    apu.utils.set_random_seeds(seed=seed)

    # Generates the data for learning
    args.tensor_grp, args.module = apu.utils.configure_dataset_args()
    config.print_configuration()
    return args


def _main(args: Namespace):
    execute_test(tg=args.tensor_grp, module=args.module)


if __name__ == '__main__':
    setup_logger(quiet_mode=False)
    _main(parse_args())
