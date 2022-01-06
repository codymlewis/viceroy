from functools import partial
import itertools
import pandas as pd

import numpy as np
import jax
import jax.flatten_util
import optax
import haiku as hk

from tqdm import trange

import hkzoo
import tenjin
import ymir

import metrics


def main(_):
    grid_results = pd.DataFrame(columns=["beta", "gamma", "0.3 mean asr", "0.3 std asr", "0.5 mean asr", "0.5 std asr"])
    print("Starting up...")
    IID = False
    DS = ymir.mp.datasets.Dataset(*tenjin.load('mnist'))
    T = 10
    ATTACK_FROM, ATTACK_TO = 0, 1
    ALG = "foolsgold"
    ADV = "onoff labelflip"
    for beta, gamma in itertools.product(np.arange(0.0, 1.1, 0.05), np.arange(0.0, 1.2, 0.05)):
        print(f"beta: {beta}, gamma: {gamma}")
        cur = {"beta": beta, "gamma": gamma}
        for acal in [0.3, 0.5]:
            print(f"Running {ALG} with {acal:.0%} {ADV} adversaries")
            net = hk.without_apply_rng(hk.transform(lambda x: hkzoo.LeNet_300_100(DS.classes, x)))

            train_eval = DS.get_iter("train", 10_000)
            test_eval = DS.get_iter("test")
            opt = optax.sgd(0.01)
            params = net.init(jax.random.PRNGKey(42), next(test_eval)[0])
            opt_state = opt.init(params)
            loss = ymir.mp.losses.cross_entropy_loss(net, DS.classes)

            A = int(T * acal)
            N = T - A
            batch_sizes = [8 for _ in range(N + A)]
            if IID:
                data = DS.fed_split(batch_sizes, ymir.mp.distributions.extreme_heterogeneous)
            else:
                data = DS.fed_split(
                    batch_sizes,
                    partial(ymir.mp.distributions.assign_classes, classes=[[(i + 1 if i >= 11 else i) % DS.classes, 11] for i in range(T)])
                )

            network = ymir.mp.network.Network()
            network.add_controller("main", server=True)
            for i in range(N):
                network.add_host("main", ymir.regiment.Scout(opt, opt_state, loss, data[i], 1))
            for i in range(A):
                c = ymir.regiment.Scout(opt, opt_state, loss, data[i + N], 1)
                ymir.regiment.adversaries.labelflipper.convert(c, DS, ATTACK_FROM, ATTACK_TO)
                ymir.regiment.adversaries.onoff.convert(c)
                network.add_host("main", c)
            controller = network.get_controller("main")
            toggler = ymir.regiment.adversaries.onoff.GradientTransform(
                params, opt, opt_state, network, ALG, controller.clients[-A:],
                max_alpha=1/N if ALG in ['fed_avg', 'std_dagmm'] else 1,
                sharp=ALG in ['fed_avg', 'std_dagmm', 'krum']
            )
            controller.add_update_transform(toggler)

            evaluator = metrics.measurer(net)
            model = getattr(ymir.garrison, ALG).Captain(params, opt, opt_state, network)
            results = metrics.create_recorder(['accuracy', 'asr'], train=True, test=True, add_evals=['attacking'])

            # Train/eval loop.
            TOTAL_ROUNDS = 3_001
            pbar = trange(TOTAL_ROUNDS)
            for round in pbar:
                if round % 10 == 0:
                    metrics.record(results, evaluator, model.params, train_eval, test_eval, {'attacking': toggler.attacking}, attack_from=ATTACK_FROM, attack_to=ATTACK_TO)
                    pbar.set_postfix({'ACC': f"{results['test accuracy'][-1]:.3f}", 'ASR': f"{results['test asr'][-1]:.3f}", 'ATT': toggler.attacking})
                model.step()
            results = metrics.finalize(results)
            cur[f"{acal} mean asr"] = results['test asr'].mean()
            cur[f"{acal} std asr"] = results['test asr'].std()
            print()
            print("=" * 150)
            print(f"Server type {ALG}, {A / (A + N):.2%} {ADV} adversaries, final accuracy: {results['test accuracy'][-1]:.3%}")
            print(metrics.tabulate(results, TOTAL_ROUNDS))
            print("=" * 150)
            print()
        grid_results = grid_results.append(cur, ignore_index=True)
    print(grid_results.to_latex())


if __name__ == "__main__":
    main()