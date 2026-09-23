from dataclasses import dataclass
from typing import Callable, Union, Literal, List, Optional
import numpy as np
from sklearn.linear_model import LogisticRegression


@dataclass
class LoadModelArgs:
    kind: Literal["load"]
    dataset: Literal["CIFAR10"]
    model_dir: str


@dataclass
class GaussiansModelArgs:
    kind: Literal['gaussians']
    centers: np.ndarray
    probs: List[float]
    n_train: int


@dataclass
class DefaultGaussiansModelArgs:
    kind: Literal['default_gaussians']
    distance: float
    probs: List[float]
    n_train: int


@dataclass
class DefaultGaussiansSeriesModelArgs:
    kind: Literal['default_gaussians_series']
    distances: List[float]
    probs: List[float]
    n_train: int


@dataclass
class TrainedGaussiansModelArgs:
    kind: Literal['trained_gaussians']
    classifier: LogisticRegression
    centers: np.ndarray
    probs: List[float]


ModelTrainingArgs = Union[LoadModelArgs, GaussiansModelArgs, DefaultGaussiansModelArgs, DefaultGaussiansSeriesModelArgs, TrainedGaussiansModelArgs]


@dataclass
class DataSamplingArgs:
    n_cal: int
    n_test: int
    probs: List[float]
    predict_scores: bool = True


@dataclass
class VectorScalingArgs:
    type: Literal['none', 'none_reduced_cal', 'only_bias', 'no_bias', 'full']
    n_vs: int


@dataclass
class ProcedureArgs:
    informative_sets: list
    informative_type: Literal['non_trivial', 'exclude_1', 'up_to_3', 'single_0_to_6', 'other']
    alpha: float
    use_cal_denominator: Optional[bool] = None
    include_threshold: Optional[bool] = None
    increase_denominator: Optional[bool] = None


@dataclass
class ExecutionArgs:
    method: Callable[
        [ModelTrainingArgs, DataSamplingArgs, VectorScalingArgs, ProcedureArgs],
        tuple[int | np.integer, int | np.integer, float | np.floating],  # incorrect_selected_test, selected_test, power
    ]
    output_folder: str
    iterations: int
    max_workers: int


@dataclass
class CalAndTestSamples:
    cal_data_classes: np.ndarray
    cal_data_samples: np.ndarray
    cal_class_scores: np.ndarray
    test_data_classes: np.ndarray
    test_data_samples: np.ndarray
    test_class_scores: np.ndarray
