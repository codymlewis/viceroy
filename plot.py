#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate the plots from a series of simulations

Author: Cody Lewis
"""

import math
from itertools import cycle
import re
import pickle

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import torch

import utils


def gen_stats(options):
    """Load the confusion matrices and calculate statistics, return stats"""
    sim_confusion_matrices = torch.load(options.result_file)
    return utils.gen_experiment_stats(sim_confusion_matrices, options)


def save_plot(options, stats, filter_fn, img_name):
    """Create and save plots as img_name based on the stats and options"""
    y_min, y_max = 0, 0
    for k, v in stats.items():
        if filter_fn(k):
            y_min = min(y_min, math.floor(v.min()))
            y_max = max(y_max, math.ceil(v.max()))
    epochs = torch.arange(options.server_epochs)
    fig, ax = plt.subplots()
    ax.set(ylim=(y_min - 0.05, y_max + 0.05))
    if options.adversaries['percent_adv'] > 0:
        if options.adversaries['optimized_look_ahead'] > 0:
            title = "{} Attack with Optimized Toggle".format(
                options.adversaries['type']
            )
            with open('toggle_record.pkl', 'rb') as f:
                toggles_rec = pickle.load(f).mean(dim=0).round().long()
            toggles = [toggles_rec[0].item()]
            for i, val in enumerate(toggles_rec[1:]):
                toggles.append((val - toggles_rec[i]).item())
            toggles = iter(toggles)
        else:
            title = "{} Attack with {} Epoch Toggle".format(
                options.adversaries['type'],
                options.adversaries['toggle_times']
            )
            toggles = cycle(options.adversaries['toggle_times'])
        if options.adversaries['delay'] is not None:
            i = options.adversaries['delay']
            on = True
            next(toggles)
        else:
            i = 0
            on = False
        nb_epochs = max(epochs)
        while i < nb_epochs:
            toggle = next(toggles)
            if on:
                rect_width = min(nb_epochs - i, toggle)
                ax.add_patch(
                    Rectangle(
                        (i, y_min),
                        rect_width,
                        abs(y_max - y_min),
                        color="red",
                        alpha=0.2
                    ),
                )
            i += toggle
            on = not on
        title = "{:.1%} {}->{} {}".format(
            options.adversaries['percent_adv'],
            options.adversaries['from'],
            options.adversaries['to'],
            title
        )
        if options.adversaries['delay'] is not None:
            title += f" and a {options.adversaries['delay']} Epoch Delay"
    else:
        title = "No Attack"
    title = f"Performance of {options.fit_fun} under {title}"
    for k in stats.keys():
        if filter_fn(k):
            ax.plot(epochs, stats[k].mean(dim=0), label=k.replace('_', ' '))
    plt.xlabel("Epochs")
    plt.ylabel("Rate")
    plt.title(title.title(), fontdict={'fontsize': 7})
    plt.legend(loc=1, fontsize=5, framealpha=0.4)
    plt.savefig(img_name, dpi=320, metadata={'comment': str(options)})
    print(f"Done. Saved plot as {img_name}")


if __name__ == '__main__':
    print("Calculating statistics and generating plots...")
    options = utils.load_options()
    stats = gen_stats(options)
    img_name = "{}_{}_{:.1%}_{}_{}_{}".format(
        options.dataset,
        options.fit_fun,
        options.adversaries['percent_adv'],
        options.adversaries['type'],
        'optimized' if options.adversaries['optimized_look_ahead'] > 0 else
        f"{options.adversaries['toggle_times']}",
        f"{d}_delay" if (d := options.adversaries['delay']) > 0
        else 'no_delay'
    ).replace(' ', '_')
    match_acc = lambda k: re.match('accuracy_\d', k)
    save_plot(options, stats, lambda k: match_acc(k) is None, f"{img_name}.png")
    save_plot(options, stats, lambda k: match_acc(k) is not None,
            f"{img_name}_accuracies.png")
