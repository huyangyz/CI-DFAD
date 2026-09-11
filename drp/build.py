from __future__ import annotations

import argparse
from pathlib import Path

from .builder import DRPBuilder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build CI-DFAD Dynamic Region Partitioning tensors."
    )
    parser.add_argument(
        "--dataset", choices=("pems", "chicago", "nyc"), default="pems"
    )
    parser.add_argument(
        "--root_path",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root containing the dataset directory.",
    )
    parser.add_argument(
        "--processed_data_path",
        type=Path,
        default=None,
        help="Optional output directory for DRP tensors.",
    )
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--num_neighbors", type=int, default=8)
    parser.add_argument(
        "--progress_every",
        type=int,
        default=100,
        help="Print progress every N timestamps; use 0 to disable.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_data_path = args.root_path / "data" / f"{args.dataset}-data"
    processed_data_path = (
        args.processed_data_path.resolve()
        if args.processed_data_path
        else args.root_path / "data" / args.dataset / "processed"
    )
    builder = DRPBuilder(
        source_data_path,
        alpha=args.alpha,
        num_neighbors=args.num_neighbors,
        progress_every=args.progress_every,
    )
    drp_path, node_path = builder.build(processed_data_path)
    print(f"Saved DRP weights to {drp_path}", flush=True)
    print(f"Saved DRP node indices to {node_path}", flush=True)


if __name__ == "__main__":
    main()
