from cifar.CIFAR10_model import build_test_cache, train_cifar10


MODEL_CONFIGS = {
    "bcd": {
        "class_names": ["bird", "cat", "dog"],
        "save_dir": "./cifar/models/bcd",
        "epochs": 30,
        "train_n": 15000,  # use all 15,000 bcd images for training
    },
    "all10": {
        "class_names": ["bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck", "airplane", "automobile"],
        "save_dir": "./cifar/models/all10",
        "epochs": 50,
        "train_n": 45000,  # 45,000 for training, 500 for validation, remaining 4,500 unused
    },
}


def train_model(model_name):
    model_dir = train_cifar10(**MODEL_CONFIGS[model_name])
    build_test_cache(model_dir)


if __name__ == "__main__":
    train_model("bcd")
    train_model("all10")