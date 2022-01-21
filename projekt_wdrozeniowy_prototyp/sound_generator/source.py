from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable

import librosa
import numpy as np


class Source(ABC):
    def __init__(
        self,
        sampling_rate: int,
    ) -> None:
        self.sampling_rate = sampling_rate

    @abstractmethod
    def draw(self) -> Iterable[np.ndarray]:
        pass


class FileSource(Source):
    def __init__(
        self,
        file_path: str,
        sampling_rate: int,
    ) -> None:
        super().__init__(sampling_rate)

        self.data_source = self._load_file(file_path)

    def _load_file(self, file_path: str) -> np.ndarray:
        file = Path(file_path)
        if not file.exists():
            raise FileNotFoundError(f'{file.absolute()} not found')

        data, _ = librosa.load(file, self.sampling_rate)
        return data

    def draw(self) -> Iterable[np.ndarray]:
        yield from self.data_source
