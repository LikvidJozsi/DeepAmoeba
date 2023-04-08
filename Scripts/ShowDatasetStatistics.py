import math
import pickle
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from AmoebaPlayGround import Amoeba

Amoeba.map_size = (10, 10)


def plot_heatmap(probs):
    fig, ax = plt.subplots()
    im = ax.imshow(probs.T)

    # We want to show all ticks...
    ax.set_xticks(np.arange(probs.shape[0]))
    ax.set_yticks(np.arange(probs.shape[1]))
    # ... and label them with the respective list entries
    # ax.set_xticklabels(farmers)
    # ax.set_yticklabels(vegetables)

    # Rotate the tick labels and set their alignment.
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
    #         rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for (x, y), value in np.ndenumerate(probs):
        text = ax.text(x, y, value, fontsize=7,
                       ha="center", va="center", color="w")

    ax.set_title("Probability distribution")
    fig.tight_layout()
    fig.show()


def calculate_entropy(sample, plot_count, max_plot_count, index):
    ent = 0.
    for i in sample.flatten():
        if i == 0:
            continue
        ent -= i * math.log(i, 2)

    '''if ent > 2 and ent < 3 and plot_count < max_plot_count:
        print(index)
        plot_heatmap(sample)
        return ent, plot_count + 1'''
    return ent, plot_count


def avg_entropy_by_reverse_turn_index(reverse_turn_indices, entropies):
    reverse_turn_index_dict = defaultdict(lambda: {"count": 0, "sum": 0})
    for index, entropy in zip(reverse_turn_indices, entropies):
        previous_sum = reverse_turn_index_dict[index]
        reverse_turn_index_dict[index] = {"count": previous_sum["count"] + 1, "sum": previous_sum["sum"] + entropy}

    sorted_keys = sorted(reverse_turn_index_dict.keys())
    reverse_turn_index_avg_list = []
    for index in sorted_keys:
        sum_for_idx = reverse_turn_index_dict[index]
        reverse_turn_index_avg_list.append(sum_for_idx["sum"] / sum_for_idx["count"])
    return sorted_keys, reverse_turn_index_avg_list


with open("../Datasets/quickstart_dataset_12x12_400_searches.p", 'rb') as file:
    dataset = pickle.load(file)
    plot_count = 0
    max_plot_count = 1
    entropies = []

    progress_bar = tqdm(total=len(dataset.board_states))
    for index, sample in enumerate(dataset.move_probabilities):
        progress_bar.update(1)
        entropy, plot_count = calculate_entropy(sample, plot_count, max_plot_count, index)
        entropies.append(entropy)

    count_win_value = 0
    for index, value in enumerate(dataset.rewards):
        if value >= 0.99:
            count_win_value += 1

    print(count_win_value)
    print(count_win_value / len(dataset.board_states))

    num_bins = 40
    n, bins, patches = plt.hist(entropies, num_bins, facecolor='blue', alpha=0.5)
    plt.show()

    indexes, avg_entropy_by_turn = avg_entropy_by_reverse_turn_index(dataset.reverse_turn_indexes, entropies)
    plt.plot(indexes, avg_entropy_by_turn)
    plt.show()
