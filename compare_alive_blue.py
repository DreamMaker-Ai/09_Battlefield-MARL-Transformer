import os.path

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    filetype = '/history/run-.-tag-mean_num_alive_blue_ratio.csv'
    filelist = [
        '01_Baseline/trial-1',
        '02_Add_LN/trial-20',
    ]
    colorlist = ['r', 'b', 'g', 'm', 'y']

    for f, c in zip(filelist, colorlist):
        ff = f + filetype
        csv_path = Path(__file__).parent / ff

        csv_df = pd.read_csv(csv_path)

        wall_time = csv_df[csv_df.columns[0]]
        step = csv_df[csv_df.columns[1]]
        num_alive_blue_ratio = csv_df[csv_df.columns[2]]

        plt.xlabel('steps')
        plt.ylabel('mean num alive blue ratio')

        plt.plot(step, num_alive_blue_ratio, linestyle='solid', color=c, alpha=0.7, linewidth=1,
                 label=f)

    # plt.yscale('log')
    plt.title('Mean num alive blue ratio vs Steps Over 50 random tests')
    plt.grid(which="both")
    plt.legend()

    savedir = Path(__file__).parent / 'history_plots'
    if not os.path.exists(savedir):
        os.mkdir(savedir)

    savename = filetype.replace('/history/run-.-tag-', '')
    savename = savename.replace('.csv', '')
    plt.savefig(str(savedir) + '/' + savename + '.png')

    plt.show()


if __name__ == '__main__':
    main()
