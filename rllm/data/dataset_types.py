import enum
from dataclasses import dataclass, field
from typing import cast


class TrainDataset:
    class Math(enum.Enum):
        # The standard American beginner competitions.
        AIME = "AIME"
        AMC = "AMC"
        # Omni math dataset
        OMNI_MATH = "OMNI_MATH"
        # Unique Olympiad problems from NUMINA
        NUMINA_OLYMPIAD = "OLYMPIAD"
        # Dan Hendrycks math
        MATH = "MATH"
        GSM8k = "GSM8K"
        STILL = "STILL"
        DEEPSCALER = "DEEPSCALER"
        DEEPSCALER_7B = "DEEPSCALER_7B"

    class MathVision(enum.Enum):
        MATHVISION = "MATHVISION"


class TestDataset:
    class Math(enum.Enum):
        AIME = "AIME"
        AMC = "AMC"
        MATH = "MATH"
        GSM8k = "GSM8k"
        MINERVA = "MINERVA"
        OLYMPIAD_BENCH = "OLYMPIAD_BENCH"

    class MathVision(enum.Enum):
        MATHVISION = "MATHVISION"


# Dataset type alias (now includes Math and MathVision)
Dataset = (
    TrainDataset.Math |
    TrainDataset.MathVision |
    TestDataset.Math |
    TestDataset.MathVision
)


@dataclass
class Problem:
    problem: str
    solution: str
    answer: str
    difficulty: float
    dataset: Dataset


@dataclass
class DatasetConfig:
    """Configuration for a dataset."""

    datasets: list[Dataset] | list[str] | str = field(default_factory=lambda: ["AMC", "AIME"])
    dataset_weights: list[float] = field(default_factory=lambda: [0.5, 0.5])
    dataloader_batch_size: int = 8

    def __post_init__(self):
        # Handle single string input
        if isinstance(self.datasets, str):
            self.datasets = [self.datasets]

        # Convert string names to Dataset enum values
        if isinstance(self.datasets[0], str):
            converted_datasets: list[Dataset] = []
            for dataset_name in self.datasets:
                dataset_found = False
                # Check TrainDataset first
                for enum_cls in [TrainDataset.Math, TrainDataset.MathVision]:
                    try:
                        dataset = enum_cls(dataset_name)
                        converted_datasets.append(cast(Dataset, dataset))
                        dataset_found = True
                        break
                    except ValueError:
                        continue

                # If not found, check TestDataset
                if not dataset_found:
                    for enum_cls in [TestDataset.Math, TestDataset.MathVision]:
                        try:
                            dataset = enum_cls(dataset_name)
                            converted_datasets.append(cast(Dataset, dataset))
                            dataset_found = True
                            break
                        except ValueError:
                            continue

                if not dataset_found:
                    raise ValueError(
                        f"Dataset {dataset_name} not found in TrainDataset or TestDataset."
                    )

            self.datasets = converted_datasets

        # Set uniform weights if not specified
        if not self.dataset_weights:
            self.dataset_weights = [1.0 / len(self.datasets)] * len(self.datasets)

        if self.dataloader_batch_size <= 0:
            raise ValueError("dataloader_batch_size must be greater than 0")

        if len(self.dataset_weights) != len(self.datasets):
            raise ValueError("Number of weights must match number of datasets")
