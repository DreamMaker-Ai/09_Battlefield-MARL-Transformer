import os.path

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


def main():
    filetype = '/history/run-.-tag-num_red_win.csv'
    filelist = [
        '01_Baseline/trial-1',
        # '02_Add_LN/trial-20',
        # '03_Big_batch/trial-30',
        # '04_Deeper_Transformer/trial-40',
        '05_Frame_stack/trial-50'
    ]
    colorlist = ['r', 'b', 'g', 'm', 'y']

    for f, c in zip(filelist, colorlist):
        ff = f + filetype
        csv_path = Path(__file__).parent / ff

        csv_df = pd.read_csv(csv_path)

        wall_time = csv_df[csv_df.columns[0]]
        diff_time = np.array(wall_time[1:]) - np.array(wall_time[:-1])
        diff_ids = np.where(diff_time > 900)

        correct_time = wall_time - wall_time[0]

        for diff_id in diff_ids[0]:
            dt = wall_time[diff_id + 1] - wall_time[diff_id]
            correct_time[diff_id + 1:] = correct_time[diff_id + 1:] - dt

        wall_time = csv_df[csv_df.columns[0]]
        step = csv_df[csv_df.columns[1]]
        num_red_win = csv_df[csv_df.columns[2]]

        plt.xlabel('learning time [hours]')
        plt.ylabel('reds win ratio')

        plt.plot(correct_time / 3600, num_red_win / 50, linestyle='solid', color=c, alpha=0.7,
                 linewidth=1, label=f)

    # plt.yscale('log')
    plt.title('Reds win ratio vs Learning time (Over 50 random tests)')
    plt.grid(which="both")
    plt.minorticks_on()
    plt.legend()

    savedir = Path(__file__).parent / 'history_plots'
    if not os.path.exists(savedir):
        os.mkdir(savedir)

    savename = filetype.replace('/history/run-.-tag-', '')
    savename = savename.replace('.csv', '')
    plt.savefig(str(savedir) + '/' + savename + '-time.png', dpi=300)

    plt.show()


if __name__ == '__main__':
    main()
