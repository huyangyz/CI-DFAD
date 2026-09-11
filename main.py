import argparse
import random
from pathlib import Path

import numpy as np
import torch

from evaluator import Evaluator
from trainer import Trainer


DATASET_CONFIGS = {
    "pems": {
        "timestamp": 12,
        "train_days": 105,
        "recent_time": 1,
        "num_feature": 12,
        "input_dim": 12,
        "time_feature": 31,
    },
    "chicago": {
        "timestamp": 2,
        "train_days": 212,
        "recent_time": 2,
        "num_feature": 4,
        "input_dim": 4,
        "time_feature": 31,
    },
    "nyc": {
        "timestamp": 2,
        "train_days": 289,
        "recent_time": 2,
        "num_feature": 4,
        "input_dim": 4,
        "time_feature": 39,
    },
}

def create_argument_parser(description="Train and evaluate CI-DFAD."):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--dataset", choices=DATASET_CONFIGS, default="chicago")
    parser.add_argument(
        "--root_path",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="CI-DFAD project root.",
    )
    parser.add_argument(
        "--processed_data_path",
        type=Path,
        default=None,
        help="Optional processed DRP directory. Defaults to data/<dataset>/processed.",
    )
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate.")
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--epoch", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lambda_G", type=float, default=500.0)
    parser.add_argument("--lambda_U", type=float, default=10.0)
    parser.add_argument("--lambda_A", type=float, default=0.05)
    parser.add_argument("--num_adj", type=int, default=9)
    parser.add_argument("--num_layer", type=int, default=2)
    parser.add_argument("--trend_time", type=int, default=7 * 24)
    parser.add_argument("--day_time", type=int, default=24)
    parser.add_argument("--mc_samples", type=int, default=100)
    parser.add_argument("--cuda_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20)
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution.")
    return parser


def parse_args():
    return create_argument_parser().parse_args()


def build_options(args):
    opt = vars(args).copy()
    config = DATASET_CONFIGS[args.dataset]
    opt.update(config)
    opt["train_time"] = config["train_days"] * config["timestamp"] * 24

    root_path = args.root_path.resolve()
    opt["root_path"] = root_path
    dataset_root = root_path / "data" / args.dataset
    output_root = root_path / "outputs" / args.dataset
    opt["source_data_path"] = root_path / "data" / f"{args.dataset}-data"
    opt["processed_data_path"] = (
        args.processed_data_path.resolve()
        if args.processed_data_path
        else dataset_root / "processed"
    )
    opt["checkpoint_path"] = output_root / "checkpoints"
    opt["result_path"] = output_root / "results"
    opt["checkpoint_path"].mkdir(parents=True, exist_ok=True)
    opt["result_path"].mkdir(parents=True, exist_ok=True)

    use_cuda = torch.cuda.is_available() and not args.cpu
    opt["cuda"] = use_cuda
    opt["device"] = torch.device(f"cuda:{args.cuda_id}" if use_cuda else "cpu")
    return opt


def set_reproducible_seed(seed, use_cuda):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    args = parse_args()
    opt = build_options(args)
    set_reproducible_seed(args.seed, opt["cuda"])

    opt["is_train"] = True
    Trainer(opt).train()

    opt["is_train"] = False
    print("evaluate...")
    Evaluator(opt).evaluate()


if __name__ == "__main__":
    main()
