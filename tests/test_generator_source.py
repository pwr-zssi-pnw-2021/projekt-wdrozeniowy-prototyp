import librosa
import numpy as np
import pytest

from projekt_wdrozeniowy_prototyp.sound_generator import (
    ArraySource,
    FileSource,
)


def test_file_loading():
    fs = FileSource('./tests/data/CantinaBand3.wav', 48000)

    assert fs.data_source is not None


@pytest.mark.xfail(raises=FileNotFoundError)
def test_missing_file():
    FileSource('./does/not/exists', 42)


def test_draw():
    fs = FileSource('./tests/data/CantinaBand3.wav', 48000)

    v = next(iter(fs.draw()))

    assert v is not None


def test_array_source():
    array = np.arange(9)
    arr_source = ArraySource(array, 42)

    loaded_array = np.array(list(arr_source.draw()))
    assert np.array_equal(array, loaded_array)


def test_array_source_resample():
    array = np.arange(9).astype(np.float32)
    initial_sr = 42
    target_sr = 23

    arr_source = ArraySource(array, initial_sr, target_sr)
    resampled_array = librosa.resample(array, initial_sr, target_sr)
    loaded_array = np.array(list(arr_source.draw()))

    assert np.array_equal(resampled_array, loaded_array)
