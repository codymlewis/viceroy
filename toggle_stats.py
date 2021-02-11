#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate stats on the toggles of a simulation

Author: Cody Lewis
"""

import matplotlib.pyplot as plt
import pandas

from pathlib import Path


def avg(a):
    return sum(a) / l if (l := len(a)) > 0 else 0


def get_stats(options, toggle_record):
    off_count = 0
    off_to_on_toggles = []
    off_to_on_count = 0
    on_to_off_toggles = []
    on_to_off_count = 0
    for row in toggle_record:
        row = row[row != -1]
        on = False
        last_time = 0
        for cell in row:
            on = not on
            if on:
                off_to_on_toggles.append(cell.item() - last_time)
                off_count += off_to_on_toggles[-1]
            else:
                on_to_off_toggles.append(cell.item() - last_time)
            last_time = cell.item()
        off_to_on_count += avg(off_to_on_toggles)
        on_to_off_count += avg(on_to_off_toggles)
        if last_time < options.server_epochs:
            if not on:
                off_count += options.server_epochs - last_time
    on_count = options.num_sims * options.server_epochs - off_count
    stats = {
        "time_on": on_count,
        "time_off": off_count,
        "off_to_on": off_to_on_count,
        "on_to_off": on_to_off_count
    }
    for k, v in stats.items():
        stats[k] = v / options.num_sims
    return stats

def write_stats(options, stats, file_name):
    if not Path(file_name).is_file():
        with open(file_name, "w") as f:
            f.write(f"ds,{','.join(stats.keys())}\n")
    with open(file_name, "a") as f:
        f.write(
            f"{options.dataset}_{options.model_params['architecture']},{','.join([str(v) for v in stats.values()])}\n"
        )


def write_results(options, toggle_record, file_name):
    stats = get_stats(options, toggle_record)
    write_stats(options, stats, file_name)

def presentable(X):
    return [x.replace('_', ' ').title() for x in X]

def make_plot(df, indices, title, stacked, img_name):
    df[indices].plot.bar(stacked=stacked, rot=0)
    plt.xticks(range(len(df)), presentable(df['ds']))
    plt.xlabel("Dataset and Model")
    plt.ylabel("Number of Epochs")
    plt.title(title)
    plt.legend(
        labels=presentable(indices[1:]),
        loc=1,
        fontsize=5,
        framealpha=0.4
    )
    plt.savefig(img_name, dpi=320)
    print(f"Done. Saved plot as {img_name}")


if __name__ == '__main__':
    df = pandas.read_csv('toggle_record.csv')
    make_plot(
        df,
        ['ds', 'time_on', 'time_off'],
        "Times Spent On and Off",
        True,
        "on_off_times.png"
    )
    make_plot(
        df,
        ['ds', 'off_to_on', 'on_to_off'],
        "Times Taken for Toggles Between States",
        False,
        "on_off_toggles.png"
    )
