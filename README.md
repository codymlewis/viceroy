# Viceroy

Source code corresponding to ["Attacks against Federated Learning Defense Systems and their Mitigation"](https://jmlr.org/papers/v23/22-0014.html) by
Cody Lewis, Vijay Varadharajan, and Nasimul Noman.

## Executing

First install the requirements:
```sh
pip install -r requirements.txt
```

You will also need to install jax as described at https://github.com/google/jax#installation

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
