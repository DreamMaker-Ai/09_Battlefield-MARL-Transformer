import json
import numpy as np
import matplotlib.pyplot as plt


def main():
    with open('test_engagement/result.json', 'r', encoding='utf-8') as f:
        json_load = json.load(f)

    R0_list = json_load['R0_list']
    B0_list = json_load['B0_list']

    R0s = np.array(R0_list)
    B0s = np.array(B0_list)

    mults = B0s / R0s
    min_mult = np.floor(np.min(mults))
    max_mult = np.ceil(np.max(mults))

    winner_list = json_load['winner']

    bins = 10
    d_mult = (max_mult - min_mult) / bins

    bin_center = np.zeros(bins)
    for j in range(bins):
        bin_center[j] = min_mult + (j + 0.5) * d_mult

    red_win = np.zeros(bins)
    blue_win = np.zeros(bins)
    no_contest = np.zeros(bins)
    draw = np.zeros(bins)

    for i, mult in enumerate(mults):
        for j in range(bins):
            if (min_mult + j * d_mult <= mult) and (mult < min_mult + (j + 1) * d_mult):
                if winner_list[i] == 'red_win':
                    red_win[j] += 1
                elif winner_list[i] == 'blue_win':
                    blue_win[j] += 1
                elif winner_list[i] == 'no_contest':
                    no_contest[j] += 1
                else:
                    draw[j] += 1

    plt.bar(bin_center, red_win, color='r')
    plt.bar(bin_center, blue_win, bottom=red_win, color='b')
    plt.bar(bin_center, no_contest, bottom=red_win, color='y')
    plt.show()


    red_win = winner_list.count('red_win')
    blue_win = winner_list.count('blue_win')
    no_contest = winner_list.count('no_contest')
    pass


if __name__ == '__main__':
    main()
