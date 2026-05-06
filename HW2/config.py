from dataclasses import dataclass, asdict
from typing import Dict, List


@dataclass
class ExperimentConfig:
    name: str
    image_size: int = 96
    num_classes: int = 10
    channels: List[int] = None
    activation: str = "relu"
    pool_type: str = "max"
    use_batchnorm: bool = False
    dropout: float = 0.0
    augment: bool = False
    optimizer: str = "sgd"
    learning_rate: float = 0.01
    weight_decay: float = 1e-4
    momentum: float = 0.9
    scheduler: str = "none"

    def to_dict(self) -> Dict:
        return asdict(self)


PRESETS: Dict[str, ExperimentConfig] = {
    "baseline": ExperimentConfig(
        name="baseline",
        channels=[32, 64, 128, 256],
        activation="relu",
        pool_type="max",
        use_batchnorm=False,
        dropout=0.0,
        augment=False,
        optimizer="sgd",
        learning_rate=0.01,
        weight_decay=1e-4,
        momentum=0.9,
        scheduler="none",
    ),
    "aug": ExperimentConfig(
        name="aug",
        channels=[32, 64, 128, 256],
        activation="relu",
        pool_type="max",
        use_batchnorm=False,
        dropout=0.0,
        augment=True,
        optimizer="sgd",
        learning_rate=0.01,
        weight_decay=1e-4,
        momentum=0.9,
        scheduler="none",
    ),
    "structure": ExperimentConfig(
        name="structure",
        channels=[32, 64, 128, 256],
        activation="leaky_relu",
        pool_type="avg",
        use_batchnorm=True,
        dropout=0.3,
        augment=True,
        optimizer="sgd",
        learning_rate=0.01,
        weight_decay=1e-4,
        momentum=0.9,
        scheduler="none",
    ),
    "optimizer": ExperimentConfig(
        name="optimizer",
        channels=[32, 64, 128, 256],
        activation="leaky_relu",
        pool_type="avg",
        use_batchnorm=True,
        dropout=0.3,
        augment=True,
        optimizer="adamw",
        learning_rate=1e-3,
        weight_decay=5e-4,
        momentum=0.9,
        scheduler="cosine",
    ),
}


def get_config(preset: str) -> ExperimentConfig:
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(PRESETS.keys())}")
    return PRESETS[preset]
