import jax
import jax.numpy as jnp


def measurer(net):
    @jax.jit
    def accuracy(params, X, y):
        predictions = net.apply(params, X)
        return jnp.mean(jnp.argmax(predictions, axis=-1) == y)

    @jax.jit
    def attack_success_rate(params, X, y, attack_from, attack_to):
        preds = jnp.argmax(net.apply(params, X), axis=-1)
        idx = y == attack_from
        return jnp.sum(jnp.where(idx, preds, -1) == attack_to) / jnp.sum(idx)
    return {'acc': accuracy, 'asr': attack_success_rate}


def create_recorder(evals, train=False, test=False, add_evals=None):
    results = dict()
    if train:
        results.update({f"train {e}": [] for e in evals})
    if test:
        results.update({f"test {e}": [] for e in evals})
    if add_evals is not None:
        results.update({e: [] for e in add_evals})
    return results


def record(results, evaluator, params, train_ds=None, test_ds=None, add_recs=None, **kwargs):
    for k, v in results.items():
        ds = train_ds if "train" in k else test_ds
        if "acc" in k:
            v.append(evaluator['acc'](params, *next(ds)))
        if ("test" in k or "train" in k) and ("asr" in k):
            v.append(evaluator['asr'](params, *next(ds), kwargs['attack_from'], kwargs['attack_to']))
    if add_recs is not None:
        for k, v in add_recs.items():
            results[k].append(v)


def finalize(results):
    for k, v in results.items():
        results[k] = jnp.array(v)
    return results


def tabulate(results, total_rounds, ri=10):
    halftime = int((total_rounds / 2) / ri)
    table = ""
    for k, v in results.items():
        table += f"[{k}] mean: {v.mean()}, std: {v.std()} [after {halftime * ri} rounds] mean {v[halftime:].mean()}, std: {v[halftime:].std()}\n"
    return table[:-1]


def csvline(ds_name, alg, adv, results, total_rounds, ri=10):
    halftime = int((total_rounds / 2) / ri)
    asr = results['test asr']
    return f"{ds_name},{alg},{adv:.2%},{results['test accuracy'][-1]},{asr.mean()},{asr.std()},{asr[halftime:].mean()},{asr[halftime:].std()}\n"