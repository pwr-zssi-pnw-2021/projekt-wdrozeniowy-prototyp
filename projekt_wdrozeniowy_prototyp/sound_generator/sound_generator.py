import time
from typing import Iterable, Optional

import numpy as np

from .preprocessor import Preprocessor, RawPreprocessor
from .source import Source


class SoundGenerator:
    """Sound generator combines sound data source and preprocessor to generate a stream of sound data chunks.

    Attributes:
        source (Source): Sound data source.
        preprocessor (Preprocessor): Sound data preprocessor. Defaults to RawPreprocessor.
        chunk_size (int): Size of generated data chunk. Defaults to 2048.
        shift (int): Additional shift for next data chunk. With shift 0, next chunk will start at current_chunk[1]. Defaults to 128.
        delay (float): Time delay between chunks, required to sync with audio
    """

    def __init__(
        self,
        source: Source,
        preprocessor: Optional[Preprocessor] = None,
        chunk_size: int = 2048,
        shift: int = 128,
    ) -> None:
        """Create new sound generator.

        Args:
            source (Source): Sound data source.
            preprocessor (Optional[Preprocessor], optional): Sound data preprocessor. If None is passed, RawPreprocessor is used. Defaults to None.
            chunk_size (int, optional): Size of generated data chunk. Defaults to 2048.
            shift (int, optional): Additional shift for next data chunk. With shift 0, next chunk will start at current_chunk[1]. Defaults to 128.
        """
        self.source = source
        self.preprocessor = preprocessor or RawPreprocessor()
        self.chunk_size = chunk_size
        self.shift = shift

        # Delay for each chunk
        self.delay = 1 / ((source.sampling_rate - chunk_size) / (shift + 1))

    def generate(self) -> Iterable[np.ndarray]:
        """Generate sound data chunks.

        Returns:
            Iterable[np.ndarray]: Sound data chunk generator.

        Yields:
            Iterator[Iterable[np.ndarray]]: Single sound data chunk.
        """
        # Preload chunk
        source_it = iter(self.source.draw())
        chunk = [next(source_it) for _ in range(self.chunk_size)]

        # Create chunks stream
        shifting = 0
        last_yield_time = None
        compensation_pool = 0.0
        for v in source_it:
            chunk.pop(0)
            chunk.append(v)

            if not shifting:
                shifting = self.shift
                chunk_arr = np.array(chunk)
                processed_chunk = self.preprocessor.process(chunk_arr)

                yield_time = time.perf_counter()
                if last_yield_time is not None:
                    time_diff = yield_time - last_yield_time
                    required_delay = self.delay - time_diff

                    if required_delay > 0:
                        if compensation_pool:
                            # if compensation required, use remaining delay to compensate
                            compensation = min(required_delay, compensation_pool)
                            required_delay -= compensation
                            compensation_pool -= compensation

                        time.sleep(required_delay)
                    else:
                        # add negative delay to compensation pool
                        compensation_pool -= required_delay

                last_yield_time = time.perf_counter()
                yield processed_chunk
            else:
                shifting -= 1
