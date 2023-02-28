import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ideadata.stock.stock_data import get_mkt_equd
from typing import Tuple

'''download data'''


def save_market_equity_daily_csv(beg_date, end_date, symbol, field, csv_path):
    # http://docs.finai.idea.edu.cn/#/ideadata/latest/ideadata.stock.html#ideadata.stock.stock_data.get_mkt_equd
    # get_mkt_equd: market_equity_daily
    df = get_mkt_equd(begin_date=beg_date, end_date=end_date, symbol=symbol, field=field)
    df.to_csv(csv_path)


def download_market_equity_daily(share_symbols: [str] = ('60000_XSHG',),
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
            pool.apply_async(func=save_market_equity_daily_csv, args=(beg_date, end_date, symbol, field, csv_path))
    pool.close()
    pool.join()


'''plot_dynamic'''


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


def plot_dynamic_share_turnover_rate():
    from shares_config import SharesShangZheng50
    data_dir_sz50 = 'share_turnover_rate_shangzheng50'
    from shares_config import SharesZhongZheng500
    data_dir_zz500 = 'share_mkt_equ_daily_zhongzheng500'

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


'''filter_shares_by_turnover_rate'''


def load_share_style_from_csv(
        data_dir: str = 'share_mkt_equ_daily_zhongzheng500',
        share_symbol: str = '600000_XSHG',
        beg_date='2020-10-31',
        end_date='2022-09-15',
) -> Tuple[float, float]:
    csv_path = f'{data_dir}/{share_symbol}.csv'

    df = pd.read_csv(csv_path)
    df_date = df['date']
    df = df[(df_date >= beg_date) & (df_date <= end_date)]

    turnover_rate = df['turnover_rate'].values.mean()
    neg_mkt_value = df['neg_mkt_value'].values.mean()
    return turnover_rate, neg_mkt_value


def load_mkt_equ_daily_from_csv(
        data_dir: str = 'share_turnover_rate_shangzheng50',
        share_symbols: tuple = ('600000_XSHG',),
        beg_date='2020-10-31',
        end_date='2022-09-15',
) -> list:
    share_info_list = []
    for share_symbol in share_symbols:
        csv_path = f'{data_dir}/{share_symbol}.csv'
        if not os.path.exists(csv_path):
            continue

        df = pd.read_csv(csv_path)
        df_date = df['date']
        df = df[(df_date >= beg_date) & (df_date <= end_date)]
        if len(df) == 0:
            continue

        turnover_rate = df['turnover_rate'].values.mean()
        neg_mkt_value = df['neg_mkt_value'].values.mean()

        share_info_list.append((share_symbol, turnover_rate, neg_mkt_value))
    return share_info_list


def demo__filter_shares_by_turnover_rate():
    # from shares_config import SharesShangZheng50
    # data_dir_sz50 = 'share_turnover_rate_shangzheng50'
    from shares_config import SharesZhongZheng500
    data_dir_zz500 = 'share_mkt_equ_daily_zhongzheng500'

    '''download'''
    beg_date = '20201031'
    end_date = '20221031'
    # for share_symbols, data_dir in ((SharesShangZheng50, data_dir_sz50),
    #                                 (SharesZhongZheng500, data_dir_zz500)):
    #     share_symbols = [share.replace('.', '_') for share in share_symbols]
    #     download_share_turnover_rate(share_symbols=share_symbols, data_dir=data_dir,
    #                                  beg_date=beg_date, end_date=end_date)

    '''read csv'''
    share_info_list = load_mkt_equ_daily_from_csv(
        data_dir=data_dir_zz500,
        share_symbols=SharesZhongZheng500,
        beg_date=beg_date, end_date=end_date,
    )

    '''filter'''
    share_info_list1 = []
    share_info_list2 = []
    for share_symbol, turnover_rate, neg_mkt_value in share_info_list:
        if (-1.7 < np.log10(turnover_rate) < -1.5) and (9.5 < np.log10(neg_mkt_value) < 9.9):
            share_info_list1.append((share_symbol, turnover_rate, neg_mkt_value))
        else:
            share_info_list2.append((share_symbol, turnover_rate, neg_mkt_value))

    print(f"len(share_info_list)    {len(share_info_list)}")
    print(f"len(share_info_list1)   {len(share_info_list1)}")
    print(repr([info[0] for info in share_info_list1]))

    '''plot'''
    share_symbols, turnover_rates, neg_mkt_values = list(zip(*share_info_list1))
    xs = np.log10(turnover_rates)
    ys = np.log10(neg_mkt_values)
    plt.scatter(xs, ys)

    share_symbols, turnover_rates, neg_mkt_values = list(zip(*share_info_list2))
    xs = np.log10(turnover_rates)
    ys = np.log10(neg_mkt_values)
    plt.scatter(xs, ys)

    plt.title('x: log10(turnover_rates)    y:np.log10(neg_mkt_values)   in ZhongZheng500')
    plt.grid()
    plt.show()


if __name__ == '__main__':
    demo__filter_shares_by_turnover_rate()
    # plot__dynamic_share_turnover_rate()
