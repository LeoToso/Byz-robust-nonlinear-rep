"""paper et al. CIFAR-10 benchmark and Byzantine-robust extension."""

from __future__ import annotations

import copy
import math
import random
from dataclasses import asdict, dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from core.aggregators import ByzantineAttack, RobustAggregator
from core.models import build_model
from core.objectives import TaskObjective, evaluate_model


def _flat(parameters: Iterable[torch.Tensor]) -> torch.Tensor:
    return torch.cat([parameter.detach().reshape(-1) for parameter in parameters])


def _set_flat(parameters: Iterable[torch.Tensor], vector: torch.Tensor) -> None:
    offset = 0
    for parameter in parameters:
        count = parameter.numel()
        parameter.data.copy_(vector[offset:offset + count].view_as(parameter))
        offset += count
    if offset != vector.numel():
        raise ValueError("Flat vector has the wrong number of elements")


def _optimizer(model: nn.Module, lr: float, momentum: float):
    weights, biases = [], []
    for name, parameter in model.named_parameters():
        (biases if "bias" in name else weights).append(parameter)
    return torch.optim.SGD(
        [{"params": weights, "weight_decay": 1e-4},
         {"params": biases, "weight_decay": 0.0}],
        lr=lr,
        momentum=momentum,
    )


def _set_trainable(model, *, backbone: bool, head: bool) -> None:
    for parameter in model.backbone.parameters():
        parameter.requires_grad_(backbone)
    for parameter in model.head.parameters():
        parameter.requires_grad_(head)


def _local_epochs(model, loader: DataLoader, epochs: int, optimizer,
                  criterion, device: torch.device) -> float:
    if epochs <= 0:
        return math.nan
    model.train()
    total_loss = 0.0
    total_samples = 0
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite local loss: {loss.item()}")
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(y)
            total_samples += len(y)
    return total_loss / total_samples


@dataclass(frozen=True)
class CIFAR10Config:
    algorithm: str = "representation_learning"
    rounds: int = 100
    population_clients: int = 100
    honest_per_round: int = 10
    byzantine_per_round: int = 0
    aggregator: str = "Average"
    attack: str = "ALIE"
    attack_tau: float = 1.5
    mimic_client: int = 0
    batch_size: int = 10
    lr: float = 0.01
    momentum: float = 0.5
    head_epochs: int = 10
    representation_epochs: int = 1
    eval_every: int = 10
    seed: int = 42
    device: str = "cpu"
    loss_type: str = "cross_entropy"


class CIFAR10Trainer:
    """paper CIFAR-10 local training with robust aggregation of deltas."""

    def __init__(self, config: CIFAR10Config,
                 train_loaders: Sequence[DataLoader],
                 test_loaders: Sequence[DataLoader]):
        self.cfg = config
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if config.algorithm not in {"representation_learning", "baseline"}:
            raise ValueError("algorithm must be representation_learning or baseline")
        if config.loss_type not in {"cross_entropy", "multiclass_ls"}:
            raise ValueError("loss_type must be cross_entropy or multiclass_ls")
        if config.population_clients != 100:
            raise ValueError("paper CIFAR-10 requires 100 honest clients")
        if len(train_loaders) != 100 or len(test_loaders) != 100:
            raise ValueError("paper CIFAR-10 requires 100 client loaders")
        if not 1 <= config.honest_per_round <= config.population_clients:
            raise ValueError("invalid honest_per_round")
        if config.byzantine_per_round == 0 and config.aggregator != "Average":
            raise ValueError("clean paper replication must use Average")
        if config.byzantine_per_round > 0 and config.aggregator == "Average":
            raise ValueError("robust extension requires a robust aggregator")

        self.train_loaders = list(train_loaders)
        self.test_loaders = list(test_loaders)
        self.device = torch.device(config.device)
        self.global_model = build_model("cifar10_paper", 64, 10).to(self.device)
        initial_head = copy.deepcopy(self.global_model.head.state_dict())
        self.client_heads = [copy.deepcopy(initial_head) for _ in range(100)]
        self.criterion = TaskObjective("classification", config.loss_type, 10)
        self.rng = np.random.default_rng(config.seed + 104729)
        self.robust_aggregator = None
        self.attack = None
        if config.byzantine_per_round:
            self.robust_aggregator = RobustAggregator(
                config.aggregator, config.byzantine_per_round)
            self.attack = ByzantineAttack(
                config.attack,
                config.byzantine_per_round,
                tau=config.attack_tau,
                epsilon=config.mimic_client,
            )

    def _aggregate(self, deltas: List[torch.Tensor],
                   weights: List[int]) -> torch.Tensor:
        if self.cfg.byzantine_per_round == 0:
            weight = torch.tensor(weights, dtype=deltas[0].dtype,
                                  device=deltas[0].device)
            weight /= weight.sum()
            return sum(w * delta for w, delta in zip(weight, deltas))
        malicious = self.attack(deltas)
        return self.robust_aggregator(deltas + malicious)

    def train_round(self) -> float:
        chosen = self.rng.choice(
            self.cfg.population_clients,
            self.cfg.honest_per_round,
            replace=False,
        )
        shared_params = (list(self.global_model.backbone.parameters())
                         if self.cfg.algorithm == "representation_learning"
                         else list(self.global_model.parameters()))
        shared_before = _flat(shared_params)
        deltas, weights, losses = [], [], []

        for client in chosen.tolist():
            local = copy.deepcopy(self.global_model)
            if self.cfg.algorithm == "representation_learning":
                local.head.load_state_dict(self.client_heads[client])
            optimizer = _optimizer(local, self.cfg.lr, self.cfg.momentum)

            if self.cfg.algorithm == "representation_learning":
                _set_trainable(local, backbone=False, head=True)
                _local_epochs(local, self.train_loaders[client],
                              self.cfg.head_epochs, optimizer,
                              self.criterion, self.device)
                self.client_heads[client] = copy.deepcopy(local.head.state_dict())
                _set_trainable(local, backbone=True, head=False)
                loss = _local_epochs(
                    local, self.train_loaders[client],
                    self.cfg.representation_epochs, optimizer,
                    self.criterion, self.device)
                local_shared = _flat(local.backbone.parameters())
            else:
                _set_trainable(local, backbone=True, head=True)
                loss = _local_epochs(
                    local, self.train_loaders[client],
                    self.cfg.representation_epochs, optimizer,
                    self.criterion, self.device)
                local_shared = _flat(local.parameters())

            deltas.append(local_shared - shared_before)
            weights.append(len(self.train_loaders[client].dataset))
            losses.append(loss)

        _set_flat(shared_params, shared_before + self._aggregate(deltas, weights))
        return float(np.mean(losses))

    def evaluate_all_clients(self) -> Tuple[float, float]:
        accuracies, losses = [], []
        for client in range(self.cfg.population_clients):
            model = copy.deepcopy(self.global_model)
            if self.cfg.algorithm == "representation_learning":
                model.head.load_state_dict(self.client_heads[client])
            metrics = evaluate_model(
                model, self.test_loaders[client], self.criterion, self.device)
            accuracies.append(metrics["metric_value"])
            losses.append(metrics["test_loss"])
        return float(np.mean(accuracies)), float(np.mean(losses))

    def run(self) -> dict:
        history, train_history = [], []
        for round_number in range(1, self.cfg.rounds + 1):
            train_loss = self.train_round()
            train_history.append({
                "round": round_number,
                "loss": train_loss,
                "train_loss": train_loss,
                "learning_rate": self.cfg.lr,
            })
            should_evaluate = (
                round_number % self.cfg.eval_every == 0 or
                round_number > self.cfg.rounds - 10
            )
            if should_evaluate:
                accuracy, test_loss = self.evaluate_all_clients()
                history.append({
                    "round": round_number,
                    "train_loss": train_loss,
                    "accuracy": accuracy,
                    "test_acc": accuracy,
                    "test_loss": test_loss,
                    "metric_name": "accuracy",
                    "metric_value": accuracy,
                    "global_acc": None,
                    "global_loss": None,
                    "global_metric_value": None,
                    "learning_rate": self.cfg.lr,
                })
                print(f"round={round_number:3d} train_loss={train_loss:.4f} "
                      f"mean_local_accuracy={accuracy:.4f}", flush=True)

        final_ten = [record for record in history
                     if record["round"] > self.cfg.rounds - 10]
        if len(final_ten) != min(10, self.cfg.rounds):
            raise RuntimeError("final-ten-round evaluation records are incomplete")
        final_metric = float(np.mean([r["accuracy"] for r in final_ten]))
        return {
            "format_version": 1,
            "benchmark": "paper21_cifar10_100_clients_2_classes",
            "dataset": "cifar10_paper",
            "population_clients": self.cfg.population_clients,
            "n_clients": (self.cfg.honest_per_round +
                          self.cfg.byzantine_per_round),
            "n_byzantine": self.cfg.byzantine_per_round,
            "alpha": None,
            "loss_type": self.cfg.loss_type,
            "algorithm": ("representation_learning"
                          if self.cfg.algorithm == "representation_learning" else "baseline"),
            "aggregator": self.cfg.aggregator,
            "attack": (self.cfg.attack
                       if self.cfg.byzantine_per_round else "None"),
            "seed": self.cfg.seed,
            "repr_dim": 64,
            "head_steps": self.cfg.head_epochs,
            "metric_name": "accuracy",
            "config": asdict(self.cfg),
            "train_history": train_history,
            "history": history,
            "final_metric": final_metric,
            "best_metric": max(r["accuracy"] for r in history),
            "reporting_rule": "unweighted client mean, averaged over final 10 rounds",
        }

    def checkpoint(self) -> dict:
        def cpu_state(module):
            return {key: value.detach().cpu()
                    for key, value in module.state_dict().items()}
        return {
            "format_version": 1,
            "benchmark": "paper21_cifar10_100_clients_2_classes",
            "config": asdict(self.cfg),
            "global_model_state_dict": cpu_state(self.global_model),
            "client_head_state_dicts": [
                {key: value.detach().cpu() for key, value in state.items()}
                for state in self.client_heads
            ],
        }
