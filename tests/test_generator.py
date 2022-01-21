import numpy as np
import pytest
from projekt_wdrozeniowy_prototyp.sound_generator import (
    FileSource,
    RawPreprocessor,
    SoundGenerator,
)


@pytest.fixture()
def source():
    fs = FileSource('./tests/data/CantinaBand3.wav', 48000)
    fs.data_source = np.arange(64)
    yield fs


@pytest.mark.parametrize(
    ['chunk_size'],
    [
        (1,),
        (3,),
        (9,),
    ],
)
def test_chunk_size(source: FileSource, chunk_size: int):
    generator = SoundGenerator(source, RawPreprocessor(), chunk_size, 0)

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
def test_shift(source: FileSource, shift: int):
    generator = SoundGenerator(source, RawPreprocessor(), 8, shift)
    gen_it = iter(generator.generate())

    chunk1 = next(gen_it)
    chunk2 = next(gen_it)

    assert chunk1[1 + shift] == chunk2[0]


def test_default_preprocessor(source: FileSource):
    generator = SoundGenerator(source)

    assert isinstance(generator.preprocessor, RawPreprocessor)
