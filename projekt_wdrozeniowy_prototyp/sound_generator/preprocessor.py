from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from librosa import feature


class Preprocessor(ABC):
    @abstractmethod
    def process(self, data: np.ndarray) -> np.ndarray:
        pass


class RawPreprocessor(Preprocessor):
    def process(self, data: np.ndarray) -> np.ndarray:
        return data


class MFCCPreprocessor(Preprocessor):
    def __init__(
        self,
        sampling_rate: int,
        features_num: Optional[int] = 20,
    ) -> None:
        self.sampling_rate = sampling_rate
        self.features_num = features_num

    def process(self, data: np.ndarray) -> np.ndarray:
        return feature.mfcc(data, sr=self.sampling_rate, n_mfcc=self.features_num)
