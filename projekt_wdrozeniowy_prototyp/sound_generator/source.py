from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable

import librosa
import numpy as np


class Source(ABC):
    """Abstract class representing sound data source."""

    def __init__(self, sampling_rate: int) -> None:
        """Create new sound data source.

        Args:
            sampling_rate (int): Source data sampling rate.
        """
        self.sampling_rate = sampling_rate

    @abstractmethod
    def draw(self) -> Iterable[np.ndarray]:
        """Yield subsequent values from loaded file.

        Returns:
            Iterable[np.ndarray]: Generator of sound data.

        Yields:
            Iterator[Iterable[np.ndarray]]: Single value from sound source.
        """
        pass


class FileSource(Source):
    def __init__(self, file_path: str, sampling_rate: int) -> None:
        """Create new sound data source using sund file.

        Uses librosa for loading the data.

        Args:
            file_path (str): Path to the sound file.
            sampling_rate (int): File sampling rate.
        """
        super().__init__(sampling_rate)

        self.data_source = self._load_file(file_path)

    def _load_file(self, file_path: str) -> np.ndarray:
        """Load file from disc.

        Args:
            file_path (str): Path to the sound file.

        Raises:
            FileNotFoundError: Sound file not found.

        Returns:
            np.ndarray: Numpy array of loaded values.
        """
        file = Path(file_path)
        if not file.exists():
            raise FileNotFoundError(f'{file.absolute()} not found')

        data, _ = librosa.load(file, self.sampling_rate)
        return data

    def draw(self) -> Iterable[np.ndarray]:
        yield from self.data_source
