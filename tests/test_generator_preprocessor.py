import librosa
import numpy as np
import pytest
from projekt_wdrozeniowy_prototyp.sound_generator import (
    MFCCPreprocessor,
    RawPreprocessor,
)


def test_raw_preprocessor():
    processor = RawPreprocessor()

    data = np.ones(5)
    processed_data = processor.process(data)

    assert np.array_equal(data, processed_data)


def test_mfcc_preprocessor():
    sampling_rate = 480000
    processor = MFCCPreprocessor(sampling_rate)

    data = np.ones(2048)
    mfcc_data = librosa.feature.mfcc(data, sampling_rate)
    processed_data = processor.process(data)

    assert np.array_equal(processed_data, mfcc_data)


@pytest.mark.parametrize(
    ['features'],
    [
        (20,),
        (15,),
        (1,),
        (100,),
    ],
)
def test_mfcc_features_num(features: int):
    processor = MFCCPreprocessor(48000, features)
    data = np.ones(2048)

    processed_data = processor.process(data)

    data_features = processed_data.shape[0]

    assert data_features == features
