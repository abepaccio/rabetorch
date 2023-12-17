import argparse

from omegaconf import OmegaConf

from rabetorch.util.config import load_config, smart_type
from rabetorch.runners.train_runner import TrainingRunner


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "config",
        type=str,
        help="Relative path to main config file from ./configs/"
    )
    p.add_argument("override", nargs="*", type=smart_type, help="Override config")

    return p.parse_args()


def main(args):
    override_dict = None
    if args.override:
        it = iter(args.override)
        override_dict = dict(zip(it, it))
    config_path = "./configs/" + args.config + ".yaml"
    cfg = load_config(config_path, override_dict)

    train_runner = TrainingRunner(cfg)
    train_runner.run_train()


if __name__ == "__main__":
    args = get_args()
    main(args)
