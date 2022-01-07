# Viceroy

A project evaluating some novel attacks against federated learning defenses.

## Executing

First install the requirements:
```sh
pip install -r requirements.txt
```

Then the main experiments can be run with:
```sh
bash experiments.sh
```

And the grid search for the on-off attack parameters can be run with:
```sh
python grid_search.py
```

## Quick recreation

The main experiment can be quickly recreated using docker with:
```sh
docker run ghcr.io/codymlewis/viceroy:latest
```