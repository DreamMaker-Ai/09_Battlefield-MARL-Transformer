import os.path

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    filetype = [
        '/history/run-.-tag-num_red_win.csv',
        '/history/run-.-tag-num_blue_win.csv',
        '/history/run-.-tag-num_no_contest.csv'
    ]
    filelist = '02_Add_LN/trial-20'

    colorlist = ['r', 'b', 'g', 'm', 'y']

    for f, c in zip(filetype, colorlist):
        ff = filelist + f
        csv_path = Path(__file__).parent / ff

        csv_df = pd.read_csv(csv_path)

        wall_time = csv_df[csv_df.columns[0]]
        step = csv_df[csv_df.columns[1]]
        prop = csv_df[csv_df.columns[2]]

        plt.xlabel('learning steps [k]')
        plt.ylabel('win ratio')

        label = f.replace('/history/run-.-tag-num_', '')
        label = label.replace('.csv', '')
        plt.plot(step / 1000, prop / 50, linestyle='solid', color=c, alpha=0.7, linewidth=1,
                 label=label)

    # plt.yscale('log')
    plt.title('Win ratio vs Learning steps (Over 50 random tests)')
    plt.grid(which="both")
    plt.minorticks_on()
    plt.legend()

    savedir = Path(__file__).parent / 'history_plots'
    if not os.path.exists(savedir):
        os.mkdir(savedir)

    savename = 'Win_looses_no-contest'
    plt.savefig(str(savedir) + '/' + savename + '.png', dpi=300)

    plt.show()


if __name__ == '__main__':
    main()
