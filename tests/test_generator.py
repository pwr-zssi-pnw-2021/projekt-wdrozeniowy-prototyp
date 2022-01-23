import time

import numpy as np
import pytest

from projekt_wdrozeniowy_prototyp.sound_generator import (
    ArraySource,
    FileSource,
    RawPreprocessor,
    SoundGenerator,
)


@pytest.fixture
def sampling_rate() -> int:
    return 48000


@pytest.fixture
def cantina_band_3s_source(sampling_rate: int) -> FileSource:
    source = FileSource('./tests/data/CantinaBand3.wav', sampling_rate)

    return source


@pytest.fixture
def array_source(sampling_rate: int) -> ArraySource:
    arr = np.arange(64)
    source = ArraySource(arr, sampling_rate)

    return source


@pytest.mark.parametrize(
    ['chunk_size'],
    [
        (1,),
        (3,),
        (9,),
    ],
)
def test_chunk_size(array_source: ArraySource, chunk_size: int):
    generator = SoundGenerator(array_source, RawPreprocessor(), chunk_size, 0)

    chunk = next(iter(generator.generate()))

    assert len(chunk) == chunk_size


@pytest.mark.parametrize(
    ['shift'],
    [
        (0,),
        (1,),
        (5,),
    ],
)
def test_shift(array_source: ArraySource, shift: int):
    generator = SoundGenerator(array_source, RawPreprocessor(), 8, shift)
    gen_it = iter(generator.generate())

    chunk1 = next(gen_it)
    chunk2 = next(gen_it)

    assert chunk1[1 + shift] == chunk2[0]


def test_default_preprocessor(array_source: ArraySource):
    generator = SoundGenerator(array_source)

    assert isinstance(generator.preprocessor, RawPreprocessor)


def test_generating_time(cantina_band_3s_source: FileSource):
    generator = SoundGenerator(cantina_band_3s_source, chunk_size=2048, shift=1024)
    target_time = 3

    time_start = time.perf_counter()
    for _ in generator.generate():
        pass
    time_end = time.perf_counter()

    assert abs((time_end - time_start) - target_time) < 10e-1


def test_generating_compensation(cantina_band_3s_source: FileSource):
    generator = SoundGenerator(cantina_band_3s_source, chunk_size=2048, shift=1024)
    target_time = 3

    time_start = time.perf_counter()
    for i, _ in enumerate(generator.generate()):
        if i == 2:
            time.sleep(0.5)
    time_end = time.perf_counter()

    assert abs((time_end - time_start) - target_time) < 10e-1
