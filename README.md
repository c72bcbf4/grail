# Graph Imitation Learning (GRAIL)

## Installation

```
uv venv --python 3.13 
uv sync
uv pip install torch-scatter -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
source .venv/bin/activate
```

## Downloading data and models
This project uses Git LFS to store models and data.
The files in both [models](models) and [data](data) are just placeholders when cloning
the repository.
You can download the original files from the online repository.

Pubchem is a large dataset and cannot be included due to storage quotas.
You can download and process the data into the correct format by running:
```
bash scripts/get_pubchem.sh
python scripts/pubchem.sh
```

## Training and evaluation 

To train a model (please make sure to download the data first):
```
source .venv/bin/activate
python main.py --config-name prod
```

To run the evaluation or interpolation (please make sure to download the models first):
```
source .venv/bin/activate
python evaluate.py 
# or
python interpolate.py
```
