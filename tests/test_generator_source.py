import pytest

from projekt_wdrozeniowy_prototyp.sound_generator import FileSource


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
