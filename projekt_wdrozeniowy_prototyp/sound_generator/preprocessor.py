from abc import ABC, abstractmethod

import numpy as np
from librosa import feature


class Preprocessor(ABC):
    """Abstract class representing data preprocessor."""

    @abstractmethod
    def process(self, data: np.ndarray) -> np.ndarray:
        """Preprocess data.

        Args:
            data (np.ndarray): Input data.

        Returns:
            np.ndarray: Processed data.
        """
        pass


class RawPreprocessor(Preprocessor):
    """Preprocessor returning raw data. No preprocessing is done."""

    def process(self, data: np.ndarray) -> np.ndarray:
        return data


class MFCCPreprocessor(Preprocessor):
    """Preprocessor generating MFCC features.

    Uses librosa for feature extraction.

    Attributes:
        sampling_rate (int): Sampling rate of input data.
        features_num (int): Number of MFCC features to generate. Defaults to 20.
    """

    def __init__(
        self,
        sampling_rate: int,
        features_num: int = 20,
    ) -> None:
        """Create MFCCPreprocessor.

        Args:
            sampling_rate (int): Sampling rate of input data.
            features_num (int, optional): Number of MFCC features to generate. Defaults to 20.
        """
        self.sampling_rate = sampling_rate
        self.features_num = features_num

    def process(self, data: np.ndarray) -> np.ndarray:
        return feature.mfcc(data, sr=self.sampling_rate, n_mfcc=self.features_num)
