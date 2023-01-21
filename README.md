# Viceroy

Source code corresponding to ["Attacks against Federated Learning Defense Systems and their Mitigation"](http://jmlr.org/papers/v24/22-0014.html) by
Cody Lewis, Vijay Varadharajan, and Nasimul Noman.

## Abstract

The susceptibility of federated learning (FL) to attacks from untrustworthy endpoints has led to the design of several defense systems. FL defense systems enhance the federated optimization algorithm using anomaly detection, scaling the updates from endpoints depending on their anomalous behavior. However, the defense systems themselves may be exploited by the endpoints with more sophisticated attacks. First, this paper proposes three categories of attacks and shows that they can effectively deceive some well-known FL defense systems. In the first two categories, referred to as on-off attacks, the adversary toggles between being honest and engaging in attacks. We analyse two such on-off attacks, label flipping and free riding, and show their impact against existing FL defense systems. As a third category, we propose attacks based on “good mouthing” and “bad mouthing”, to boost or diminish influence of the victim endpoints on the global model. Secondly, we propose a new federated optimization algorithm, Viceroy, that can successfully mitigate all the proposed attacks. The proposed attacks and the mitigation strategy have been tested on a number of different experiments establishing their effectiveness in comparison with other contemporary methods. The proposed algorithm has also been made available as open source. Finally, in the appendices, we provide an induction proof for the on-off model poisoning attack, and the proof of convergence and adversarial tolerance for the new federated optimization algorithm.

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
