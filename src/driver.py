from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from pathlib import Path

from generate_results import calculate_results
from learner import CalibratedLearner, APU_Learner
from puc_learner import PUcLearner
from apu import config, setup_logger
from apu.datasets import synthetic
from apu.datasets.types import BaseFFModule
import apu.utils


def parse_args() -> Namespace:
    r""" Parse, checks, and refactors the input arguments"""
    args = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
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
    if args.d:
        apu.utils.set_debug_mode(seed=1)

    # Generates the data for learning
    args.tensor_grp, args.module = apu.utils.configure_dataset_args()
    config.print_configuration()
    return args


def _main(args: Namespace):
    if config.DATASET.is_synthetic():
        cal_module = synthetic.Module()
    else:
        cal_module = BaseFFModule(x=args.tensor_grp.p_x, num_hidden_layers=config.NUM_SIGMA_LAYERS)
    cal_learner = CalibratedLearner(gamma=config.GAMMA, prior=config.TRAIN_PRIOR,
                                    base_module=cal_module)
    cal_learner.fit(args.tensor_grp)
    if config.DATASET.is_synthetic():
        apu.utils.log_decision_boundary(cal_learner.block, name="Sigma")

    apu_learners = APU_Learner(base_module=args.module, sigma=cal_learner, rho_vals=[0.5],
                               bias_priors=args.bias_priors)
    apu_learners.fit(args.tensor_grp)

    priors = [config.TRAIN_PRIOR]
    if args.bias_priors:
        scalars = [0.8, 1.2]
        priors.extend(scale * config.TRAIN_PRIOR for scale in scalars)

    pucs = []
    for prior in priors:
        puc_learner = PUcLearner(prior=prior)
        puc_learner.fit(args.tensor_grp)
        pucs.append(puc_learner)

    calculate_results(args.tensor_grp, apu_learners, pucs)

    if config.DATASET.is_synthetic():
        stem = args.config_file.stem.replace("_", "-")
        for name, block in apu_learners.blocks():
            plot_path = apu.utils.PLOTS_DIR / f"{name.lower()}_centroids_{stem}.png"
            apu.utils.plot_centroids(plot_path, args.tensor_grp,
                                     decision_boundary=block.module.decision_boundary())


if __name__ == '__main__':
    setup_logger(quiet_mode=False)
    _main(parse_args())
