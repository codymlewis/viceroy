import pickle
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def make_plot(filename, results, attacking=None):
    rounds = range(0, 5_001, 10)
    fig, ax = plt.subplots()
    ax.set_ylim(0, 1)
    if attacking is not None:
        for (start, width) in attacking:
            ax.add_patch(
                Rectangle(
                    (start * 10, 0),
                    width * 10,
                    1,
                    color="red",
                    alpha=0.1
                ),
            )
    for k, v in results.items():
        if k != 'attacking':
            plt.plot(rounds, v, label=k.title().replace('Asr', 'ASR'))
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
    ax.legend(fontsize=7, ncol=4, loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=False, shadow=False, frameon=False)
    plt.xlabel('Rounds')
    plt.ylabel('Rate')
    plt.savefig(filename, dpi=320)


def attack_points(attacking):
    state = False
    points = []
    for i, a in enumerate(attacking):
        a = not a
        if a and not state:
            start = i
            width = 0
            state = True
        if a:
            width += 1
        if state and not a:
            points.append((start, width))
            state = False
    if state:
        points.append((start, width + 1))
    return points


if __name__ == "__main__":
    with open('results.pkl', 'rb') as f:
        results = pickle.load(f)
    fn = "plot.pdf"
    make_plot(fn, results, attacking=attack_points(results['attacking']))
    print(f"Save plot to {fn}")