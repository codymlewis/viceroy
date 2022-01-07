import sys
import os
import argparse
from functools import partial
import pandas as pd
import logging

import numpy as np
import jax
import jax.flatten_util
import jax.numpy as jnp
import optax
import haiku as hk

from tqdm import trange

import hkzoo
import tenjin
import ymir

import metrics


def main(args):
    adv_percent = [0.1, 0.3, 0.5, 0.8]
    if os.path.exists("results.pkl"):
        final_results = pd.read_pickle("results.pkl")
    else:
        final_results = pd.DataFrame(columns=["algorithm", "attack", "dataset"] + [f"{p} mean asr" for p in adv_percent] + [f"{p} std asr" for p in adv_percent])
    VICTIM = 0
    T = 100
    ALG = args.alg
    ADV = args.attack
    DATASET = args.dataset
    aper = args.aper
    print("Starting up...")
    DS = ymir.mp.datasets.Dataset(*tenjin.load(DATASET))
    if DATASET == 'kddcup99':
        ATTACK_FROM, ATTACK_TO = 0, 11
    else:
        ATTACK_FROM, ATTACK_TO = 0, 1
    cur = {"algorithm": ALG, "attack": ADV, "dataset": DATASET}
    rng = np.random.default_rng(0)
    print(f"Running {ALG} on {DATASET} with {aper:.0%} {ADV} adversaries")
    if DATASET == 'cifar10':
        net = hk.without_apply_rng(hk.transform(lambda x: hkzoo.ConvLeNet(DS.classes, x)))
    else:
        net = hk.without_apply_rng(hk.transform(lambda x: hkzoo.LeNet_300_100(DS.classes, x)))

    train_eval = DS.get_iter("train", 10_000, rng=rng)
    test_eval = DS.get_iter("test", rng=rng)
    opt = optax.sgd(0.01)
    params = net.init(jax.random.PRNGKey(42), next(test_eval)[0])
    opt_state = opt.init(params)
    loss = ymir.mp.losses.cross_entropy_loss(net, DS.classes)

    A = int(T * aper)
    N = T - A
    batch_sizes = [8 for _ in range(N + A)]
    if DATASET != 'kddcup99':
        data = DS.fed_split(batch_sizes, partial(ymir.mp.distributions.lda, alpha=0.5 if ALG in ['contra', 'foolsgold', 'viceroy'] else 1000), rng)
    else:
        data = DS.fed_split(
            batch_sizes,
            partial(ymir.mp.distributions.assign_classes, classes=[[(i + 1 if i >= 11 else i) % DS.classes, 11] for i in range(T)]),
            rng
        )

    network, toggler = create_network(N, A, ADV, params, opt, opt_state, loss, data, batch_sizes, DS, DATASET, ALG, ATTACK_FROM, ATTACK_TO, VICTIM)

    evaluator = metrics.measurer(net)

    if "backdoor" in ADV:
        test_eval = DS.get_iter(
            "test",
            map=partial({
                "mnist": ymir.regiment.adversaries.backdoor.mnist_backdoor_map,
                "cifar10": ymir.regiment.adversaries.backdoor.cifar10_backdoor_map,
                "kddcup99": ymir.regiment.adversaries.backdoor.kddcup99_backdoor_map
            }[DATASET], ATTACK_FROM, ATTACK_TO, no_label=True)
        )

    if ALG == "krum":
        model = getattr(ymir.garrison, ALG).Captain(params, opt, opt_state, network, rng, clip=A)
    elif ALG == "contra":
        model = getattr(ymir.garrison, ALG).Captain(params, opt, opt_state, network, rng, k=N)
    else:
        model = getattr(ymir.garrison, ALG).Captain(params, opt, opt_state, network, rng)

    results = metrics.create_recorder(['accuracy', 'asr'], train=True, test=True, add_evals=['attacking'])
    results["asr"] = []

    # Train/eval loop.
    TOTAL_ROUNDS = 5000
    for round in (pbar := trange(TOTAL_ROUNDS)):
        alpha, all_grads = model.step()
        if round % 1 == 0:
            attacking = toggler.attacking if toggler else True
            record_metrics(results, evaluator, alpha, all_grads, model.params, train_eval, test_eval, ADV, ALG, A, attacking, ATTACK_FROM, ATTACK_TO, VICTIM)
            pbar.set_postfix({'ACC': f"{results['test accuracy'][-1]:.3f}", 'ASR': f"{results['asr'][-1]:.3f}", 'ATT': attacking})
    results = metrics.finalize(results)
    cur[f"{aper} mean asr"] = results['asr'].mean()
    cur[f"{aper} std asr"] = results['asr'].std()
    print()
    print("=" * 150)
    print(f"Server type {ALG}, Dataset {DATASET}, {A / (A + N):.2%} {ADV} adversaries, final accuracy: {results['test accuracy'][-1]:.3%}")
    print(metrics.tabulate(results, TOTAL_ROUNDS))
    print("=" * 150)
    print()
    idx = (final_results['algorithm'] == ALG) & (final_results['attack'] == ADV) & (final_results['dataset'] == DATASET)
    if idx.any():
        final_results.loc[idx, cur.keys()] = cur.values()
    else:
        final_results = final_results.append(cur, ignore_index=True)
    write_results(final_results, "results.pkl")


@jax.jit
def euclid_dist(a, b):
    return jnp.sqrt(jnp.sum((a - b)**2, axis=-1))


def unzero(x):
    return max(x, sys.float_info.epsilon)


def write_results(results, fn):
    results.to_pickle(fn)
    print(f"Written results to {fn}")


def create_network(num_honest, num_adv, attack, params, opt, opt_state, loss, data, batch_sizes, ds, dataset, alg, att_from, att_to, victim):
    if alg == "krum":
        server_kwargs = {"clip": num_adv}
    elif alg == "contra":
        server_kwargs = {"k": num_honest}
    else:
        server_kwargs = {}
    network = ymir.mp.network.Network()
    network.add_controller("main", server=True)
    for i in range(num_honest):
        network.add_host("main", ymir.regiment.Scout(opt, opt_state, loss, data[i], 1))
    for i in range(num_adv):
        c = ymir.regiment.Scout(opt, opt_state, loss, data[i + num_honest], batch_sizes[i + num_honest])
        if "labelflip" in attack:
            ymir.regiment.adversaries.labelflipper.convert(c, ds, att_from, att_to)
        elif "backdoor" in attack:
            ymir.regiment.adversaries.backdoor.convert(c, ds, dataset, att_from, att_to)
        elif "freerider" in attack:
            ymir.regiment.adversaries.freerider.convert(c, "delta", params)
        if "onoff" in attack:
            ymir.regiment.adversaries.onoff.convert(c)
        network.add_host("main", c)
    controller = network.get_controller("main")
    if "scaling" in attack:
        controller.add_update_transform(ymir.regiment.adversaries.scaler.GradientTransform(params, opt, opt_state, network, alg, num_adv, **server_kwargs))
    if "mouther" in attack:
        controller.add_update_transform(ymir.regiment.adversaries.mouther.GradientTransform(num_adv, victim, attack))
    if "onoff" not in attack:
        toggler = None
    else:
        if len(server_kwargs) > 0:
            server_kwargs["timer"] = True
        toggler = ymir.regiment.adversaries.onoff.GradientTransform(
            params, opt, opt_state, network, alg, controller.clients[-num_adv:],
            max_alpha=1/num_honest if alg in ['fed_avg', 'std_dagmm'] else 1,
            sharp=alg in ['fed_avg', 'std_dagmm', 'krum'],
            **server_kwargs
        )
        controller.add_update_transform(toggler)
    return network, toggler


def record_metrics(results, evaluator, alpha, all_grads, params, train_eval, test_eval, attack, alg, num_adv, attacking, attack_from, attack_to, victim):
    metrics.record(results, evaluator, params, train_eval, test_eval, {'attacking': attacking}, attack_from=attack_from, attack_to=attack_to)
    if "freerider" in attack:
        if attacking:
            if alg == "krum":
                results['asr'].append(alpha[-num_adv:].mean())
            else:
                results['asr'].append(jnp.minimum(alpha[-num_adv:].mean() / (1 / (alpha > 0).sum()), 1))
        else:
            results['asr'].append(0.0)
    elif "mouther" in attack:
        if (alpha[-num_adv:] < 0.0001).all():
            asr = -1 if alpha[victim] < 0.0001 else -2
        else:
            theta = jax.flatten_util.ravel_pytree(ymir.garrison.sum_grads(all_grads))[0]
            vicdel = euclid_dist(jax.flatten_util.ravel_pytree(all_grads[victim])[0], theta)
            if "good" in attack:
                numerator = min(euclid_dist(jnp.array([jax.flatten_util.ravel_pytree(g)[0] for g in all_grads]), theta))
                asr = unzero(numerator) / unzero(vicdel)
            else:
                asr = unzero(vicdel) / unzero(max(euclid_dist(jnp.array([jax.flatten_util.ravel_pytree(g)[0] for g in all_grads]), theta)))
        results['asr'].append(asr)
    else:
        results["asr"].append(results["test asr"][-1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the viceroy main experiment.')
    parser.add_argument('--alg', type=str, default="fedavg", help='Algorithm to use')
    parser.add_argument('--attack', type=str, default="onoff labelflip", help='Attack to use')
    parser.add_argument('--dataset', type=str, default="mnist", help='Dataset to use')
    parser.add_argument('--aper', type=float, default=0.1, help='Percentage of adversaries in the network')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    main(args)