#!/usr/bin/env python3
"""Run the School Exam experiments reported in the paper."""
import argparse, itertools, json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.run_experiment import run_experiment

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    p.add_argument("--algorithms", nargs="+", choices=["baseline", "representation_learning"], default=["baseline", "representation_learning"])
    p.add_argument("--aggregators", nargs="+", default=["NNM+Krum", "NNM+TrMean"])
    p.add_argument("--attacks", nargs="+", default=["ALIE", "Mimic"])
    p.add_argument("--device", default="cpu")
    p.add_argument("--data_dir", default="data")
    p.add_argument("--output_dir", default="results/school")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    for algorithm, aggregator, attack, seed in itertools.product(a.algorithms, a.aggregators, a.attacks, a.seeds):
        path = Path(a.output_dir) / algorithm / aggregator / attack / f"seed_{seed}.json"
        checkpoint = path.with_suffix(".pt")
        if path.exists() and checkpoint.exists() and not a.overwrite:
            print(f"skip {path}", flush=True); continue
        print(f"run {path}", flush=True)
        result = run_experiment(dataset="school", alpha=1.0, n_clients=144, n_byzantine=5,
            algorithm=algorithm, aggregator=aggregator, attack=attack, loss_type="least_squares",
            repr_dim=64, head_steps=20, lr=0.05, lr_head=0.01, momentum=0.9,
            rounds=500, batch_size=32, active_honest_clients=20, eval_every=10,
            data_dir=a.data_dir, device=a.device, seed=seed, checkpoint_path=str(checkpoint))
        payload = result.to_dict(); payload.update(honest_per_round=20, byzantine_per_round=5)
        path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_suffix(".tmp")
        with temporary.open("w") as stream: json.dump(payload, stream, indent=2, allow_nan=False)
        os.replace(temporary, path)

if __name__ == "__main__": main()
