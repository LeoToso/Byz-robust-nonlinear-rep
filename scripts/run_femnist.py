#!/usr/bin/env python3
"""Run the clean paper FEMNIST calibration or its Byzantine extension."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.femnist_experiment import (FEMNISTConfig, FEMNISTTrainer,
                                  load_femnist_partition)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition", default="data/femnist_partition")
    parser.add_argument("--output_dir", default="results/femnist")
    parser.add_argument("--mode", choices=["clean", "robust"], required=True)
    parser.add_argument("--algorithms", nargs="+", choices=["baseline", "representation_learning"],
                        default=["baseline", "representation_learning"])
    parser.add_argument("--aggregators", nargs="+",
                        default=["NNM+TrMean", "NNM+Krum"])
    parser.add_argument(
        "--attacks", nargs="+",
        choices=["ALIE", "Mimic", "SignFlipping", "InnerProductManipulation"],
        default=["ALIE", "Mimic"],
    )
    parser.add_argument(
        "--attack_tau", type=float, default=1.5,
        help="Attack factor for IPM and ALIE (default: 1.5)",
    )
    parser.add_argument(
        "--mimic_client", type=int, default=0,
        help="Zero-based honest client position copied by Mimic (default: 0)",
    )
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[42, 123, 456])
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--honest_per_round", nargs="+", type=int, default=[10, 20, 50])
    parser.add_argument(
        "--loss_type",
        choices=["cross_entropy", "multiclass_ls"],
        default="cross_entropy",
    )
    parser.add_argument("--byzantine_per_round", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def output_path(root: str, config: FEMNISTConfig) -> Path:
    attack = config.attack if config.byzantine_per_round else "None"
    return (Path(root) / f"honest_{config.honest_per_round}" /
            config.algorithm / config.aggregator / attack /
            f"seed_{config.seed}.json")


def main():
    args = parse_args()
    train_sets, test_sets, metadata = load_femnist_partition(args.partition)
    if metadata["n_clients"] != 150 or metadata["n_classes"] != 10:
        raise ValueError("This runner requires the 150-client, 10-class paper partition")

    if args.mode == "clean":
        grid = itertools.product(args.honest_per_round, args.algorithms, ["Average"], ["SignFlipping"], args.seeds)
        byzantine = 0
    else:
        grid = itertools.product(args.honest_per_round, args.algorithms, args.aggregators, args.attacks, args.seeds)
        byzantine = args.byzantine_per_round

    for honest_per_round, algorithm, aggregator, attack, seed in grid:
        config = FEMNISTConfig(
            algorithm=algorithm, rounds=args.rounds,
            honest_per_round=honest_per_round,
            byzantine_per_round=byzantine, aggregator=aggregator, attack=attack,
            attack_tau=args.attack_tau,
            mimic_client=args.mimic_client,
            eval_every=args.eval_every, seed=seed, device=args.device,
            loss_type=args.loss_type,
            official_softmax_ce=False,
        )
        path = output_path(args.output_dir, config)
        checkpoint_path = path.with_suffix(".pt")
        if path.exists() and checkpoint_path.exists() and not args.overwrite:
            print(f"skip {path}")
            continue
        print(f"run {path}", flush=True)
        trainer = FEMNISTTrainer(config, train_sets, test_sets)
        result = trainer.run()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        with temporary.open("w") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
        os.replace(temporary, path)
        torch.save(trainer.checkpoint(), checkpoint_path)


if __name__ == "__main__":
    main()
