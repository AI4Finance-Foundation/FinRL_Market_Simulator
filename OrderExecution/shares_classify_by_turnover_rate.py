import os
import time

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ideadata.stock.stock_data import get_mkt_equd

"""
shares_classify_by_turnover_rate.py
2022-12-05 09:24:44
"""


def get_mkt_equd_and_save(beg_date, end_date, symbol, field, csv_path):
    # http://docs.finai.idea.edu.cn/#/ideadata/latest/ideadata.stock.html#ideadata.stock.stock_data.get_mkt_equd
    df = get_mkt_equd(begin_date=beg_date, end_date=end_date, symbol=symbol, field=field)
    df.to_csv(csv_path)


def download_share_turnover_rate(share_symbols: [str] = ('60000_XSHG',),
                                 data_dir: str = 'share_turnover_rate_shangzheng50',
                                 beg_date='20201031',
                                 end_date='20201031'):
    share_symbols = [share.replace('.', '_') for share in share_symbols]
    from multiprocessing import Pool
    pool = Pool(processes=16)

    os.makedirs(data_dir, exist_ok=True)
    for share_symbol in share_symbols:
        csv_path = f'{data_dir}/{share_symbol}.csv'
        if not os.path.exists(csv_path):
            print(f"| {csv_path}")
            symbol = [share_symbol.replace('_', '.'), ]
            field = ['symbol', 'date', 'open_px', 'close_px', 'high_px', 'low_px',
                     'volume', 'deal_amt', 'neg_mkt_value', 'mkt_value', 'chg_pct', 'turnover_rate']

            # get_mkt_equd_and_save(beg_date, end_date, symbol, field, csv_path)
            pool.apply_async(func=get_mkt_equd_and_save, args=(beg_date, end_date, symbol, field, csv_path))
    pool.close()
    pool.join()


def plot_share_turnover_rate(data_dir: str = 'share_turnover_rate_shangzheng50',
                             colors: [tuple] = ((0.5, 0.0, 1.0, 1.0),),
                             beg_date='2020-10-31',
                             end_date='2022-09-15',
                             ):
    file_names = os.listdir(data_dir)
    file_names = sorted(file_names)
    for file_name, color in zip(file_names, colors):
        share_symbol = file_name[:-4]
        csv_path = f'{data_dir}/{share_symbol}.csv'
        if not os.path.exists(csv_path):
            continue

        df = pd.read_csv(csv_path)
        df_date = df['date']
        df = df[(df_date >= beg_date) & (df_date <= end_date)]
        if len(df) == 0:
            # print(csv_path)
            continue

        xs = np.log10(df['turnover_rate'].clip(lower=1e-4))
        ys = np.log10(df['neg_mkt_value'].clip(lower=1e-4))
        plt.scatter(xs, ys, color=color, label=share_symbol)
        plt.plot(xs, ys, color=color, label=share_symbol)


def demo__share_turnover_rate():
    from shares_config import SharesShangZheng50
    data_dir_sz50 = 'share_turnover_rate_shangzheng50'
    from shares_config import SharesZhongZheng500
    data_dir_zz500 = 'share_turnover_rate_zhongzheng500'

    '''download'''
    # beg_date = '20201031'
    # end_date = '20221031'
    # for share_symbols, data_dir in ((SharesShangZheng50, data_dir_sz50),
    #                                 (SharesZhongZheng500, data_dir_zz500)):
    #     share_symbols = [share.replace('.', '_') for share in share_symbols]
    #     download_share_turnover_rate(share_symbols=share_symbols, data_dir=data_dir,
    #                                  beg_date=beg_date, end_date=end_date)

    '''plot'''
    from matplotlib.cm import get_cmap
    color_map = get_cmap('rainbow')
    colors_sz50 = [color_map(i / 50) for i in range(len(SharesShangZheng50))]
    colors_zz500 = [color_map(i / 500) for i in range(len(SharesZhongZheng500))]

    plt.ion()

    dates = [[f"{year:04}-{month:02}-{day:02}" for day in range(1, 29)]
             for year in (2021, 2022) for month in range(1, 13)][:-3]
    dates = sum(dates, [])
    beg_end_dates = [(dates[i], dates[i + 14]) for i in range(0, len(dates) - 14, 4)]
    for beg_date, end_date in beg_end_dates:
        plt.cla()
        plt.title(f"From {beg_date} to {end_date}")
        plt.grid()
        plot_share_turnover_rate(data_dir=data_dir_sz50, colors=colors_sz50,
                                 beg_date=beg_date, end_date=end_date)
        # plot_share_turnover_rate(data_dir=data_dir_zz500, colors=colors_zz500,
        #                          beg_date=beg_date, end_date=end_date)
        plt.pause(0.1)

    plt.ioff()
    plt.legend()
    plt.show()


if __name__ == '__main__':
    demo__share_turnover_rate()
