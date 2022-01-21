from typing import Iterable, Optional

import numpy as np

from .preprocessor import Preprocessor, RawPreprocessor
from .source import Source


class SoundGenerator:
    def __init__(
        self,
        source: Source,
        preprocessor: Optional[Preprocessor] = None,
        chunk_size: int = 2048,
        shift: int = 128,
    ) -> None:
        self.source = source
        self.preprocessor = preprocessor or RawPreprocessor()
        self.chunk_size = chunk_size
        self.shift = shift

    def generate(self) -> Iterable[np.ndarray]:
        # Preload chunk
        source_it = iter(self.source.draw())
        chunk = [next(source_it) for _ in range(self.chunk_size)]

        # Create chunks stream
        shifting = 0
        for v in source_it:
            chunk.pop(0)
            chunk.append(v)

            if not shifting:
                shifting = self.shift
                chunk_arr = np.array(chunk)
                processed_chunk = self.preprocessor.process(chunk_arr)

                yield processed_chunk
            else:
                shifting -= 1
