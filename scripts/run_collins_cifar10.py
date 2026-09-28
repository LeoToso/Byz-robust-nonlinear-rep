#!/usr/bin/env python3
"""Run Collins CIFAR-10 or its Byzantine client-scaling extension."""

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

from core.collins_cifar10 import CollinsCIFAR10Config, CollinsCIFAR10Trainer
from core.datasets import load_cifar10_collins


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--output_dir", default="results/cifar10_collins")
    parser.add_argument("--mode", choices=["clean", "robust"], required=True)
    parser.add_argument("--algorithms", nargs="+", choices=["fedavg", "fedrep"],
                        default=["fedavg", "fedrep"])
    parser.add_argument("--aggregators", nargs="+",
                        default=["NNM+TrMean", "NNM+Krum"])
    parser.add_argument("--attacks", nargs="+", choices=["ALIE", "Mimic"],
                        default=["ALIE", "Mimic"])
    parser.add_argument("--loss_types", nargs="+",
                        choices=["cross_entropy", "multiclass_ls"],
                        default=["cross_entropy", "multiclass_ls"])
    parser.add_argument("--attack_tau", type=float, default=1.5)
    parser.add_argument("--mimic_client", type=int, default=0)
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[42, 123, 456, 789, 1024])
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--honest_per_round", nargs="+", type=int,
                        default=[10, 20, 50])
    parser.add_argument("--byzantine_per_round", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def output_path(root: str, config: CollinsCIFAR10Config) -> Path:
    attack = config.attack if config.byzantine_per_round else "None"
    return (Path(root) / config.loss_type /
            f"honest_{config.honest_per_round}" / config.algorithm /
            config.aggregator / attack / f"seed_{config.seed}.json")


def main():
    args = parse_args()
    if args.rounds != 100:
        print("[WARN] Collins CIFAR-10 uses 100 rounds.", flush=True)

    if args.mode == "clean":
        grid = itertools.product(
            args.loss_types, args.honest_per_round, args.algorithms,
            ["Average"], ["ALIE"], args.seeds)
        byzantine = 0
    else:
        grid = itertools.product(
            args.loss_types, args.honest_per_round, args.algorithms,
            args.aggregators, args.attacks, args.seeds)
        byzantine = args.byzantine_per_round

    for loss_type, honest, algorithm, aggregator, attack, seed in grid:
        config = CollinsCIFAR10Config(
            algorithm=algorithm,
            rounds=args.rounds,
            honest_per_round=honest,
            byzantine_per_round=byzantine,
            aggregator=aggregator,
            attack=attack,
            attack_tau=args.attack_tau,
            mimic_client=args.mimic_client,
            eval_every=args.eval_every,
            seed=seed,
            device=args.device,
            loss_type=loss_type,
        )
        path = output_path(args.output_dir, config)
        checkpoint_path = path.with_suffix(".pt")
        if path.exists() and checkpoint_path.exists() and not args.overwrite:
            print(f"skip {path}", flush=True)
            continue

        # Rebuild the seeded randomized Collins partition for each trial.
        train, test, _ = load_cifar10_collins(
            100, data_dir=args.data_dir, batch_size=config.batch_size,
            seed=seed)
        print(f"run {path}", flush=True)
        trainer = CollinsCIFAR10Trainer(config, train, test)
        result = trainer.run()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        with temporary.open("w") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
        os.replace(temporary, path)
        torch.save(trainer.checkpoint(), checkpoint_path)


if __name__ == "__main__":
    main()
