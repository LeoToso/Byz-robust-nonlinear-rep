"""Dataset construction for the reported CIFAR-10 and School Exam results."""
import os
from typing import Dict, List
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

def train_test_split_indices(indices, test_ratio=0.2, seed=42):
    rng = np.random.default_rng(seed); indices = np.asarray(indices); rng.shuffle(indices)
    n_test = max(1, int(len(indices) * test_ratio))
    return indices[n_test:].tolist(), indices[:n_test].tolist()

def cifar10_class_pairs(seed: int = 42):
    """Balanced assignment of two distinct CIFAR-10 classes to 100 clients."""
    rng = np.random.default_rng(seed)
    labels = rng.permutation(np.repeat(np.arange(10, dtype=np.int64), 20))
    pairs = labels.reshape(100, 2).copy()
    while True:
        loops = np.flatnonzero(pairs[:, 0] == pairs[:, 1])
        if not len(loops): break
        i = int(loops[0])
        candidates = np.flatnonzero((pairs[:, 0] != pairs[:, 1]) &
                                    (pairs[:, 1] != pairs[i, 0]) &
                                    (pairs[:, 0] != pairs[i, 0]))
        if not len(candidates):
            labels = rng.permutation(labels); pairs = labels.reshape(100, 2).copy(); continue
        j = int(rng.choice(candidates)); pairs[i, 1], pairs[j, 1] = pairs[j, 1], pairs[i, 1]
    return [tuple(map(int, pairs[index])) for index in rng.permutation(len(pairs))]

def load_paper_cifar10(n_clients, alpha=None, data_dir="data", batch_size=10, seed=42):
    """Build 100 clients with two classes, 500 train and 100 test images each."""
    if n_clients != 100: raise ValueError("The CIFAR-10 population has exactly 100 clients")
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((.5,.5,.5),(.5,.5,.5))])
    train_ds = datasets.CIFAR10(data_dir, train=True, download=True, transform=tfm)
    test_ds = datasets.CIFAR10(data_dir, train=False, download=True, transform=tfm)
    rng = np.random.default_rng(seed)
    train_by_class = [np.flatnonzero(np.asarray(train_ds.targets) == k) for k in range(10)]
    test_by_class = [np.flatnonzero(np.asarray(test_ds.targets) == k) for k in range(10)]
    for indices in train_by_class + test_by_class: rng.shuffle(indices)
    train_pos, test_pos, train_loaders, test_loaders = [0]*10, [0]*10, [], []
    for labels in cifar10_class_pairs(seed):
        train_idx, test_idx = [], []
        for label in labels:
            train_idx.extend(train_by_class[label][train_pos[label]:train_pos[label]+250])
            test_idx.extend(test_by_class[label][test_pos[label]:test_pos[label]+50])
            train_pos[label] += 250; test_pos[label] += 50
        train_loaders.append(DataLoader(Subset(train_ds, train_idx), batch_size=batch_size, shuffle=True))
        test_loaders.append(DataLoader(Subset(test_ds, test_idx), batch_size=batch_size, shuffle=False))
    return train_loaders, test_loaders, DataLoader(test_ds, batch_size=256, shuffle=False)

def _school_task_ranges(task_indexes, n_samples):
    idx = np.asarray(task_indexes).astype(int).squeeze()
    if idx.ndim == 2 and 2 in idx.shape:
        pairs = idx if idx.shape[1] == 2 else idx.T
        if pairs.min() >= 1: pairs = pairs - np.array([1, 0])
        return [(int(a), int(b)) for a, b in pairs]
    idx = idx.reshape(-1)
    if idx[-1] == n_samples and idx[0] != 0:
        return list(zip(np.r_[0, idx[:-1]].astype(int), idx.astype(int)))
    if idx[0] == 1: idx = idx - 1
    if idx[0] != 0: idx = np.r_[0, idx]
    if idx[-1] != n_samples: idx = np.r_[idx, n_samples]
    return [(int(idx[i]), int(idx[i + 1])) for i in range(len(idx) - 1)]

def load_school(n_clients, alpha, data_dir="data", batch_size=32, seed=42):
    """Load naturally partitioned School Exam regression clients."""
    from scipy.io import loadmat
    from sklearn.preprocessing import StandardScaler
    from urllib.request import urlretrieve
    school_dir = os.path.join(data_dir, "school"); os.makedirs(school_dir, exist_ok=True)
    mat_path = os.path.join(school_dir, "school.mat")
    if not os.path.exists(mat_path):
        urlretrieve("https://raw.githubusercontent.com/jiayuzhou/MALSAR/master/data/school.mat", mat_path)
    raw = loadmat(mat_path)
    X = np.asarray(raw["X"], dtype=np.float32); y = np.asarray(raw["Y"], dtype=np.float32).reshape(-1)
    if X.shape[0] != len(y) and X.shape[1] == len(y): X = X.T
    ranges = _school_task_ranges(raw["task_indexes"], len(y))
    if len(ranges) < n_clients: raise ValueError(f"Requested {n_clients} schools, found {len(ranges)}")
    rng = np.random.default_rng(seed); chosen = sorted(rng.choice(len(ranges), n_clients, replace=False).tolist())
    splits, all_train = [], []
    for client_id in chosen:
        start, end = ranges[client_id]
        train_rel, test_rel = train_test_split_indices(range(end-start), seed=seed+client_id)
        pair = ([start+j for j in train_rel], [start+j for j in test_rel]); splits.append(pair); all_train.extend(pair[0])
    X = StandardScaler().fit(X[all_train]).transform(X).astype(np.float32)
    y_mean, y_std = float(y[all_train].mean()), float(y[all_train].std()) or 1.0
    y = ((y-y_mean)/y_std).astype(np.float32); DATASET_META["school"]["dim"] = int(X.shape[1])
    class SchoolDataset(Dataset):
        def __init__(self, features, targets): self.X, self.y = torch.from_numpy(features), torch.from_numpy(targets)
        def __len__(self): return len(self.y)
        def __getitem__(self, index): return self.X[index], self.y[index]
    full = SchoolDataset(X, y); client_train, client_test, global_test_idx = [], [], []
    for train_idx, test_idx in splits:
        client_train.append(DataLoader(Subset(full, train_idx), batch_size=batch_size, shuffle=True))
        client_test.append(DataLoader(Subset(full, test_idx), batch_size=batch_size, shuffle=False)); global_test_idx.extend(test_idx)
    return client_train, client_test, DataLoader(Subset(full, global_test_idx), batch_size=256, shuffle=False)

DATASET_META: Dict[str, Dict] = {
    "cifar10_paper": {"n_classes": 10, "in_ch": 3, "img": 32, "dim": 3*32*32},
    "school": {"n_classes": 1, "in_ch": 1, "img": 0, "dim": 27, "task": "regression"},
}

def get_loaders(dataset, n_clients, alpha, data_dir="data", batch_size=64, seed=42, use_leaf=False):
    if dataset == "cifar10_paper": return load_paper_cifar10(n_clients, alpha, data_dir, batch_size, seed)
    if dataset == "school": return load_school(n_clients, alpha, data_dir, batch_size, seed)
    raise ValueError(f"Unknown dataset: {dataset}")

def heterogeneity_score(client_loaders, n_classes, task="classification") -> float:
    if task == "regression":
        means = [float(torch.cat([y.float().reshape(-1) for _, y in loader]).mean()) for loader in client_loaders]
        n = len(means); return float(sum(abs(means[i]-means[j]) for i in range(n) for j in range(i+1,n))/max(n*(n-1)/2,1))
    freqs = []
    for loader in client_loaders:
        counts = np.zeros(n_classes)
        for _, y in loader:
            for label in y.numpy(): counts[int(label)] += 1
        freqs.append(counts/(counts.sum()+1e-9))
    freqs = np.asarray(freqs); n = len(freqs)
    return float(sum(np.abs(freqs[i]-freqs[j]).sum() for i in range(n) for j in range(i+1,n))/max(n*(n-1)/2,1))
