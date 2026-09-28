import numpy as np
import pytest

torch = pytest.importorskip("torch")

from torch.utils.data import DataLoader, TensorDataset

from core.collins_cifar10 import _local_epochs
from core.datasets import collins_cifar10_class_pairs
from core.models import build_model


def test_collins_cifar10_random_shards_are_balanced_and_distinct():
    pairs = collins_cifar10_class_pairs(seed=42)
    labels = np.asarray(pairs)
    assert labels.shape == (100, 2)
    assert np.all(labels[:, 0] != labels[:, 1])
    assert np.bincount(labels.reshape(-1), minlength=10).tolist() == [20] * 10
    assert pairs == collins_cifar10_class_pairs(seed=42)
    assert pairs != collins_cifar10_class_pairs(seed=123)


def test_collins_cifar10_model_has_released_five_layer_split():
    model = build_model("cifar10_collins", repr_dim=999, n_classes=10)
    assert model.backbone.repr_dim == 64
    assert model.backbone.conv1.in_channels == 3
    assert model.backbone.conv1.out_channels == 64
    assert model.backbone.conv2.in_channels == 64
    assert model.backbone.conv2.out_channels == 64
    assert model.backbone.fc1.in_features == 64 * 5 * 5
    assert model.backbone.fc1.out_features == 120
    assert model.backbone.fc2.in_features == 120
    assert model.backbone.fc2.out_features == 64
    assert model.head.fc.in_features == 64
    assert model.head.fc.out_features == 10


def test_local_epochs_sweep_every_batch_each_epoch():
    x = torch.randn(12, 2)
    y = torch.randn(12, 1)
    loader = DataLoader(TensorDataset(x, y), batch_size=3, shuffle=False)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    class CountingMSE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, prediction, target):
            self.calls += 1
            return torch.nn.functional.mse_loss(prediction, target)

    criterion = CountingMSE()
    _local_epochs(model, loader, epochs=2, optimizer=optimizer,
                  criterion=criterion, device=torch.device("cpu"))
    assert criterion.calls == 2 * len(loader)
