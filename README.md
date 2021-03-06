# Projekt Naukowo-Wdrożeniowy

### Temat: Real Time Speech Emotion Recognition
### Prototyp

![badge](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/greenpp/945bb032814d966ad859f99e23f7fe18/raw/badge.json)

## How to install

### Requirements
- [Python 3.9](https://www.python.org/)
- [Poetry](https://python-poetry.org/)
- [Pre-commit](https://pre-commit.com/)

### Installation
Install python packages
```sh
poetry install
```
Install pre-commit hooks
```sh
pre-commit install
```

## How to run
```sh
streamlit run projekt_wdrozeniowy_prototyp/interface.py
```

## Documentation
Documentation can be built with make
```sh
make html
```

## Test coverage report
```sh
pytest --cov-report html --cov=projekt_wdrozeniowy_prototyp tests
```
