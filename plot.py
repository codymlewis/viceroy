#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from itertools import cycle

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import torch

import utils


if __name__ == '__main__':
    print("Calculating statistics and generating plot...")
    options = utils.load_options()
    sim_confusion_matrices = torch.load(options.result_file)
    stats = utils.gen_experiment_stats(sim_confusion_matrices, options)
    epochs = torch.arange(len(stats['accuracy'][0]))
    fig, ax = plt.subplots()
    if options.adversaries['percent_adv'] > 0:
        if options.adversaries['type'] == "on off":
            title = "On-Off Attack with {} Epoch Toggle".format(
                options.adversaries['toggle_times']
            )
            i = 0
            toggles = cycle(options.adversaries['toggle_times'])
            on = False
            nb_epochs = max(epochs)
            while i < nb_epochs:
                toggle = next(toggles)
                if on:
                    rect_width = min(nb_epochs - i, toggle)
                    ax.add_patch(
                        Rectangle(
                            (i, 0),
                            rect_width,
                            1,
                            color="red",
                            alpha=0.2
                        ),
                    )
                i += toggle
                on = not on
        else:
            title = f"{options.adversaries['type']} Attack"
        title = "{}% {}->{} {}".format(
            options.adversaries['percent_adv'] * 100,
            options.adversaries['from'],
            options.adversaries['to'],
            title
        )
    else:
        title = "No Attack"
    title = f"Performance of {options.fit_fun} under {title}"
    for k in stats.keys():
        ax.plot(epochs, stats[k].mean(dim=0), label=k.replace('_', ' '))
    plt.xlabel("Epochs")
    plt.ylabel("Rate")
    plt.title(title.title(), fontdict={'fontsize': 8})
    plt.legend(loc=1, fontsize=5, framealpha=0.4)
    img_name = "{}_{}_{}_{}.png".format(
        options.dataset,
        options.fit_fun,
        options.adversaries['percent_adv'] * 100,
        options.adversaries['type']
    ).replace(' ', '_')
    plt.savefig(img_name, dpi=320, metadata={'comment': str(options)})
    print(f"Done. Saved plot as {img_name}")
