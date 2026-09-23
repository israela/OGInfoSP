import os, json, numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from simulation.interfaces import CalAndTestSamples

CIFAR10_LABELS = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


class Net(nn.Module):
    def __init__(self, in_channels, num_classes, dropout_prob=0.3):
        super().__init__()
        def make_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        self.stem = make_block(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)  # 32 → 16
        self.b2 = make_block(32, 64)
        self.pool2 = nn.MaxPool2d(2)  # 16 → 8
        self.b3 = make_block(64, 128)
        self.pool3 = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x, return_features=False):
        x = self.stem(x)
        x = self.pool1(x)
        x = self.b2(x)
        x = self.pool2(x)
        x = self.b3(x)
        x = self.pool3(x).flatten(1)
        features = F.relu(x)
        features = self.dropout(features)
        logits = self.fc(features)
        return (logits, features) if return_features else logits


class NNClassifier:
    def __init__(self, model, batch_size=128, n_epochs=10, lr=1e-3, weight_decay=5e-4):
        self.model = model
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.loss_function = nn.CrossEntropyLoss(label_smoothing=0.05)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.n_iter_no_change = 10
        self.tol = 1e-4


    def fit(self, train_dataset, val_dataset=None, device=None):
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2, pin_memory=True) if val_dataset else None

        best_accuracy = 0.0
        epochs_without_improvement = 0
        for _ in range(self.n_epochs):
            self.model.train()
            for x, y in train_loader:
                x, y = x.to(device), y.long().to(device)
                self.optimizer.zero_grad()
                loss = self.loss_function(self.model(x), y)
                loss.backward()
                self.optimizer.step()

            if not val_loader:
                continue
            self.model.eval()
            correct = total = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.long().to(device)
                    predicted_classes = self.model(x).softmax(-1).argmax(1)
                    correct += (predicted_classes == y).sum().item()
                    total += y.size(0)
            validation_accuracy = correct / max(1,total)
            if validation_accuracy > best_accuracy + self.tol:
                best_accuracy = validation_accuracy
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement > self.n_iter_no_change:
                    break


def _norm():
    return transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))


def _build_dataset(class_names, is_training, normalize=True):
    global_class_ids = [CIFAR10_LABELS.index(name) for name in class_names]
    global_to_local = {global_class_id: local_class_id for local_class_id, global_class_id in enumerate(global_class_ids)}
    tf = transforms.Compose([transforms.ToTensor(), _norm()] if normalize else [transforms.ToTensor()])
    dataset = datasets.CIFAR10("./data", train=is_training,  download=True, transform=tf)
    idx = [i for i in range(len(dataset)) if dataset[i][1] in global_class_ids]
    subset = RemapSubset(dataset, idx, global_to_local) if is_training else Subset(dataset, idx)
    return subset, global_class_ids


def _labels_of_subset(dataset):
    return np.array([dataset[i][1] for i in range(len(dataset))], dtype=int)



def _remap_global_to_local(global_labels, global_class_ids):
    global_to_local = {global_class_id: local_class_id for local_class_id, global_class_id in enumerate(global_class_ids)}
    return np.array([global_to_local[int(global_class_id)] for global_class_id in global_labels], dtype=np.int64)


class RemapSubset(torch.utils.data.Dataset):
    def __init__(self, base, indices, global_to_local):
        self.base = base
        self.indices = np.asarray(indices, int)
        self.global_to_local = global_to_local
    def __len__(self): return len(self.indices)
    def __getitem__(self, i):
        x, y = self.base[int(self.indices[i])]
        return x, self.global_to_local[int(y)]


def train_cifar10(
    class_names=None,        # all 10 classes by default; otherwise, a subset such as ["bird", "cat", "dog"]
    save_dir="./cnn_model",
    train_n=None,            # all available images from the selected classes by default
    epochs=10,               # non-paper default
    lr=1e-3,                 # adam learning rate
    batch_size=128,
    seed=2020,
    normalize=True
):
    os.makedirs(save_dir, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    if class_names is None:
        class_names = CIFAR10_LABELS[:]
    train_dataset, global_class_ids = _build_dataset(class_names, is_training=True, normalize=normalize)
    train_labels = _labels_of_subset(train_dataset)

    # choose an approximately balanced training subset when train_n is specified
    rng = np.random.default_rng(seed)
    if train_n is None or train_n >= len(train_dataset):
        train_subset = train_dataset
        val_subset = None
    else:
        K = len(global_class_ids)
        train_counts = rng.multinomial(train_n, np.ones(K) / K)
        indices_by_class = {class_id: np.where(train_labels == class_id)[0] for class_id in range(K)}
        selected_indices = []
        for class_id, count in enumerate(train_counts):
            count = min(count, len(indices_by_class[class_id]))
            selected_indices.extend(rng.choice(indices_by_class[class_id], size=count, replace=False))
        train_subset = Subset(train_dataset, np.array(selected_indices, dtype=int))
        # reserve a small validation subset from the remaining training indices for early stopping
        remaining_indices = np.setdiff1d(np.arange(len(train_dataset)), selected_indices)
        val_size = min(500, len(remaining_indices) // 10)
        val_subset = Subset(train_dataset, rng.choice(remaining_indices, size=val_size, replace=False)) if val_size > 0 else None

    K = len(global_class_ids)
    model = Net(in_channels=3, num_classes=K)
    clf = NNClassifier(model=model, batch_size=batch_size, n_epochs=epochs, lr=lr)
    clf.fit(train_subset, val_subset)

    torch.save({"state_dict": model.state_dict(), "num_classes": K}, os.path.join(save_dir, "model.pt"))
    with open(os.path.join(save_dir, "meta.json"), "w") as f:
        json.dump({"class_names": class_names, "wanted_ids": global_class_ids, "normalize": bool(normalize)}, f)
    return save_dir


@torch.no_grad()
def _forward_probs_feats(model, dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)
    prob_batches, feature_batches, label_batches = [], [], []
    for x,y in loader:
        x = x.to(device)
        logits, features = model(x, return_features=True)
        prob_batches.append(logits.softmax(-1).cpu())
        feature_batches.append(features.cpu())
        label_batches.append(y.clone())
    return (torch.cat(label_batches).numpy().astype(np.int64),
            torch.cat(feature_batches).numpy().astype(np.float32),
            torch.cat(prob_batches).numpy().astype(np.float32))


def _load_model(model_dir):
    checkpoint = torch.load(os.path.join(model_dir, "model.pt"), map_location="cpu")
    meta = json.load(open(os.path.join(model_dir, "meta.json")))
    model = Net(in_channels=3, num_classes=checkpoint["num_classes"])
    model.load_state_dict(checkpoint["state_dict"])
    return model, meta


def build_test_cache(model_dir):
    """compute features and probabilities for all test images of the trained classes and save them to disk"""
    checkpoint = torch.load(os.path.join(model_dir, "model.pt"), map_location="cpu")
    meta = json.load(open(os.path.join(model_dir, "meta.json")))
    class_names = meta["class_names"]
    global_class_ids = meta["wanted_ids"]
    normalize = bool(meta.get("normalize", True))

    # build the test split for the selected classes
    test_dataset, _ = _build_dataset(class_names, is_training=False, normalize=normalize)

    # load the trained model
    model = Net(in_channels=3, num_classes=checkpoint["num_classes"])
    model.load_state_dict(checkpoint["state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    # run inference once over the selected test images
    loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())
    prob_batches, feature_batches, global_label_batches = [], [], []
    with torch.no_grad():
        for x, global_labels in loader:
            x = x.to(device)
            logits, features = model(x, return_features=True)
            prob_batches.append(logits.softmax(-1).cpu().numpy())
            feature_batches.append(features.cpu().numpy())
            global_label_batches.append(global_labels.numpy())
    all_probs = np.concatenate(prob_batches).astype(np.float32)                 # (N_test, K)
    all_features = np.concatenate(feature_batches).astype(np.float32)           # (N_test, D=128)
    all_global_labels = np.concatenate(global_label_batches).astype(np.int64)   # (N_test,)

    np.savez_compressed(
        os.path.join(model_dir, "test_cache.npz"),
        probs=all_probs,
        feats=all_features,
        labels_global=all_global_labels,
        wanted_ids=np.array(global_class_ids, dtype=np.int64),
        class_names=np.array(class_names),
        normalize=np.array([normalize], dtype=np.bool_)
    )


def sample_cifar10(probs, n_cal, n_test, model_dir="./cnn_model", seed=None):
    cache_path = os.path.join(model_dir, "test_cache.npz")
    if not os.path.exists(cache_path):
        build_test_cache(model_dir)

    cached_test_data = np.load(cache_path, allow_pickle=True)
    global_class_ids = cached_test_data["wanted_ids"].tolist()  # column order in all_probs
    class_names = cached_test_data["class_names"].tolist()
    all_probs = cached_test_data["probs"]                    # (N, K)
    all_features = cached_test_data["feats"]                 # (N, D)
    all_global_labels = cached_test_data["labels_global"]    # (N,)
    K = len(global_class_ids)

    if probs is None:
        probs = np.ones(K) / K
    probs = np.asarray(probs, float)
    assert probs.shape == (K,) and np.isclose(probs.sum(), 1.0), "probs must sum to 1 over K"

    # group test indices by global class ID
    indices_by_global_class = {
        global_class_id: np.where(all_global_labels == global_class_id)[0]
        for global_class_id in global_class_ids
    }

    rng = np.random.default_rng() if seed is None else np.random.default_rng(seed)
    cal_class_counts = rng.multinomial(n_cal, probs)
    test_class_counts = rng.multinomial(n_test, probs)

    # sample without replacement
    cal_index_parts = []
    test_index_parts = []
    for global_class_id, cal_count, test_count in zip(global_class_ids, cal_class_counts, test_class_counts):
        total_count = cal_count + test_count
        if total_count > len(indices_by_global_class[global_class_id]):
            class_name = CIFAR10_LABELS[global_class_id]
            raise ValueError(
                f"Cannot sample {total_count} distinct '{class_name}' images; "
                f"the test cache contains only {len(indices_by_global_class[global_class_id])}."
            )
        sampled_indices = rng.choice(indices_by_global_class[global_class_id], size=total_count, replace=False)
        cal_index_parts.append(sampled_indices[:cal_count])
        test_index_parts.append(sampled_indices[cal_count:])

    cal_indices = np.concatenate(cal_index_parts) if n_cal > 0 else np.array([], dtype=int)
    test_indices = np.concatenate(test_index_parts) if n_test > 0 else np.array([], dtype=int)

    # shuffle the order within each split
    if cal_indices.size:  cal_indices  = cal_indices[rng.permutation(cal_indices.size)]
    if test_indices.size: test_indices = test_indices[rng.permutation(test_indices.size)]

    # select the sampled rows from the cache
    cal_class_scores = all_probs[cal_indices]       # (n_cal, K)
    test_class_scores = all_probs[test_indices]     # (n_test, K)
    cal_features = all_features[cal_indices]        # (n_cal, D)
    test_features = all_features[test_indices]      # (n_test, D)

    # remap global class IDs to local class IDs in 0, ..., K - 1
    global_to_local = {
        global_class_id: local_class_id
        for local_class_id, global_class_id in enumerate(global_class_ids)
    }
    cal_local_labels = np.array(
        [global_to_local[int(global_class_id)] for global_class_id in all_global_labels[cal_indices]],
        dtype=np.int64,
    )
    test_local_labels = np.array(
        [global_to_local[int(global_class_id)] for global_class_id in all_global_labels[test_indices]],
        dtype=np.int64,
    )

    return CalAndTestSamples(cal_local_labels, cal_features, cal_class_scores, test_local_labels, test_features, test_class_scores)