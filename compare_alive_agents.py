import os.path

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    filetype = [
        '/history/run-.-tag-mean_num_alive_red_ratio.csv',
        '/history/run-.-tag-mean_num_alive_blue_ratio.csv',
    ]
    filelist = '01_Baseline/trial-1'
    # filelist = '02_Add_LN/trial-20'
    # filelist = '03_Big_batch/trial-30'
    # filelist = '04_Deeper_Transformer/trial-40'
    # filelist = '05_Frame_stack/trial-50'

    colorlist = ['r', 'b', 'g', 'm', 'y']

    for f, c in zip(filetype, colorlist):
        ff = filelist + f
        csv_path = Path(__file__).parent / ff

        csv_df = pd.read_csv(csv_path)

        wall_time = csv_df[csv_df.columns[0]]
        step = csv_df[csv_df.columns[1]]
        prop = csv_df[csv_df.columns[2]]

        plt.xlabel('learning steps [k]')
        plt.ylabel('average alive agents ratio')

        label = f.replace('/history/run-.-tag-mean_', '')
        label = label.replace('.csv', '')
        plt.plot(step / 1000, prop, linestyle='solid', color=c, alpha=0.7, linewidth=1,
                 label=label)

    # plt.yscale('log')
    plt.title('Average num of alive agents (Over 50 random tests)')
    plt.grid(which="both")
    plt.minorticks_on()
    plt.legend()

    savedir = Path(__file__).parent / 'history_plots'
    if not os.path.exists(savedir):
        os.mkdir(savedir)

    savename = 'Alive agents'
    plt.savefig(str(savedir) + '/' + savename + '.png', dpi=300)

    plt.show()


if __name__ == '__main__':
    main()
