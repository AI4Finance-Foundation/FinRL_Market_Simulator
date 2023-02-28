import os
import torch
import pandas as pd

TEN = torch.Tensor


def _add_norm(tensor_dicts, n_days=5):
    pass

    '''get avg and the vam(the mean of var)'''
    price_avgs = torch.stack([d['price'].mean(dim=0) for d in tensor_dicts])
    price_vams = torch.stack([(d['price'] ** 2).mean(dim=0, keepdim=True) for d in tensor_dicts])
    volume_avgs = torch.stack([d['volume'].mean(dim=0) for d in tensor_dicts])
    volume_vams = torch.stack([(d['volume'] ** 2).mean(dim=0, keepdim=True) for d in tensor_dicts])
    value_avgs = torch.stack([d['value'].mean(dim=0) for d in tensor_dicts])
    value_vams = torch.stack([(d['value'] ** 2).mean(dim=0, keepdim=True) for d in tensor_dicts])

    '''save avg and std for normalization using previous day data'''

    def get_std(_vam, _avg):
        return torch.sqrt(_vam - _avg ** 2)

    for i_day in range(len(tensor_dicts)):
        i, j = (0, n_days) if i_day <= n_days else (i_day - n_days, i_day)
        tensor_dict = tensor_dicts[i_day]

        price_avg = price_avgs[i:j].mean()
        price_std = get_std(price_vams[i:j].mean(), price_avg)
        tensor_dict['price_norm'] = price_avg.item(), price_std.item()

        volume_avg = volume_avgs[i:j].mean()
        volume_std = get_std(volume_vams[i:j].mean(), volume_avg)
        tensor_dict['volume_norm'] = volume_avg.item(), volume_std.item()

        value_avg = value_avgs[i:j].mean()
        value_std = get_std(value_vams[i:j].mean(), value_avg)
        tensor_dict['value_norm'] = value_avg.item(), value_std.item()
    return tensor_dicts


def _add_share_style(tensor_dicts, n_days=5, data_dir='./share_mkt_equ_daily_zhongzheng500'):
    from shares_config import SharesZhongZheng500
    from shares_classify import download_market_equity_daily

    '''get mkt_equity_daily_df'''
    if not os.path.exists(data_dir):
        download_market_equity_daily(
            share_symbols=SharesZhongZheng500,
            data_dir=data_dir,
            beg_date='20101031',
            end_date='20201031',
        )

    share_symbols = [tensor_dict['share_symbol'] for tensor_dict in tensor_dicts]
    share_symbols = list(set(share_symbols))
    assert len(share_symbols) == 1
    share_symbol = share_symbols[0]
    csv_path = f'{data_dir}/{share_symbol}.csv'
    mkt_equity_daily_df = pd.read_csv(csv_path)
    mkt_equity_daily_df = mkt_equity_daily_df.set_index('date')

    turnover_rates = []
    neg_mkt_values = []
    for tensor_dict in tensor_dicts:
        trade_date = tensor_dict['trade_date']
        turnover_rates.append(mkt_equity_daily_df.loc[trade_date, 'turnover_rate'])
        neg_mkt_values.append(mkt_equity_daily_df.loc[trade_date, 'neg_mkt_value'])
    turnover_rates = torch.tensor(turnover_rates)
    neg_mkt_values = torch.tensor(neg_mkt_values)

    for i_day in range(len(tensor_dicts)):
        i, j = (0, n_days) if i_day <= n_days else (i_day - n_days, i_day)
        tensor_dict = tensor_dicts[i_day]

        tensor_dict['turnover_rate'] = turnover_rates[i:j].mean()
        tensor_dict['neg_mkt_value'] = neg_mkt_values[i:j].mean()
    return tensor_dicts


def get_trade_dates(beg_date: str = '2022.09.01', end_date: str = '2022.09.15') -> [str]:
    from ideadata.stock.trade_calendar import TradeCalendar
    cal_df = TradeCalendar().get_trade_cal(beg_date, end_date)
    cal_df = cal_df[cal_df.is_open == 1]

    trade_dates = [item.date for item in cal_df.itertuples()]
    return trade_dates


def get_share_dicts_by_day(share_dir='./shares_data_by_day', share_symbol='000525_XSHE',
                           beg_date='2022-09-01', end_date='2022-09-30',
                           n_levels=5, device=None):
    if device is None:
        gpu_id = 0
        device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")

    '''convert csv to tensor by day'''
    if beg_date is None:
        trade_dates = [csv_name[:-4]
                       for csv_name in sorted(os.listdir(share_dir))
                       if csv_name[-4:] == '.csv']
    else:
        try:
            trade_dates = get_trade_dates(beg_date=beg_date, end_date=end_date)
        except ModuleNotFoundError:
            trade_dates = [csv_name[:-4] for csv_name in sorted(os.listdir(share_dir))
                           if csv_name[-4:] == '.csv']
            beg_id = len([trade_date for trade_date in trade_dates if trade_date < beg_date])
            end_id = len([trade_date for trade_date in trade_dates if trade_date <= end_date])
            trade_dates = trade_dates[beg_id:end_id]

    tensor_dicts = []
    for trade_date in trade_dates:
        csv_path = f"{share_dir}/{share_symbol}/{trade_date}.csv"
        tensor_dict = csv_to_tensor_dict(csv_path=csv_path, device=device, n_levels=n_levels)
        tensor_dict['trade_date'] = trade_date
        tensor_dict['share_symbol'] = share_symbol
        tensor_dicts.append(tensor_dict)
    return tensor_dicts


def csv_to_tensor_dict(csv_path: str, device, n_levels: int = 5):
    df = pd.read_csv(csv_path)

    def get_tensor(ary):
        return torch.tensor(ary, dtype=torch.float32, device=device)

    """找出超过3秒的时间间隔，并进行插值处理，将它们都转化为3秒的间隔。例如：6秒插值后变成两个3秒。9秒变三个3秒
    Find time intervals exceeding 3 seconds, and perform interpolation to convert them to 3-second intervals. 
    For example, inserting a value into 6 seconds of data results in two 3-second data intervals, 
    and inserting two values into 9 seconds of data results in three 3-second intervals.
    """
    '''找出超过3秒的时间间隔
    Find time intervals exceeding 3 seconds
    '''
    df['UpdateTime'] = pd.to_datetime(df['UpdateTime'])  # 将字符串转换为 Pandas 时间戳对象
    df['UpdateTime'] = df['UpdateTime'].apply(lambda x: x.timestamp())  # 将时间戳对象转换为 POSIX 时间戳（浮点数形式）
    time_stamp = df['UpdateTime'].values
    time_diffs = time_stamp[1:] - time_stamp[:-1]

    '''新建需要插值的所有row，更新 UpdateTime 用于排序
    Create new rows for all data points requiring interpolation and update the UpdateTime for sorting purposes
    '''
    df_list = []
    for i, time_diff in enumerate(time_diffs):
        if 6 <= time_diff <= 1800:
            for j in range(int(time_diff) // 3 - 1):
                # df.loc[i, 'UpdateTime'] = df.loc[i, 'UpdateTime'] - j * 3
                df_list.append(df.iloc[i])
    '''有时候 收盘最后时刻的3秒快照没有更新，上面的for循环就无法补全需要插值的row，因此使用下面的代码补全'''
    for j in range(4800 + 2 - len(df) - len(df_list)):
        # df.loc[-1, 'UpdateTime'] = df.loc[-1, 'UpdateTime'] + j * 3 + 3
        df_list.append(df.iloc[-1])
    add_df = pd.concat(df_list, axis=1, ignore_index=True).T

    '''重建 pd.concat 操作后 column 丢失的 dtype'''
    for column in df.columns:
        add_df[column] = add_df[column].astype(df[column].dtype)

    '''通过concat添加到原本的df里，根据 UpdateTime 进行排序，完成插值
    Add the new rows to the original dataframe using concat, 
    and sort the dataframe based on UpdateTime to complete the interpolation
    '''
    df = pd.concat((df, add_df))
    df.sort_values('UpdateTime')
    df.reset_index(drop=True)
    assert df.shape == (4800 + 2, 27)

    """get data for building tensor_dict"""
    '''get delta volume'''
    volume = get_tensor(df['Volume'].values)
    volume[1:] = torch.diff(volume, n=1)  # delta volume
    torch.nan_to_num_(volume, nan=0)

    '''get delta turnover (value)'''
    value = get_tensor(df['turnover'].values)
    value[1:] = torch.diff(value, n=1)  # delta turnover (value)
    torch.nan_to_num_(value, nan=0)

    '''get last price'''
    price = get_tensor(df['LastPrice'].fillna(method='ffill').values)  # last price
    price[price == 0] = price[price > 0][-1]

    '''fill nan in ask_prices and ask_volumes'''
    ask_prices = get_tensor(df[[f'AskPrice{i}' for i in range(1, n_levels + 1)]].values).T
    assert ask_prices.shape == (n_levels, len(df))
    for i in range(n_levels):
        prev_price = ask_prices[i - 1] if i > 0 else price
        ask_prices[i] = fill_zero_and_nan_with_tensor(ask_prices[i], prev_price + 0.01)
    ask_volumes = get_tensor(df[[f'AskVolume{i}' for i in range(1, n_levels + 1)]].values).T
    torch.nan_to_num_(ask_volumes, nan=0)

    '''fill nan in bid_prices and bid_volumes'''
    bid_prices = get_tensor(df[[f'BidPrice{i}' for i in range(1, n_levels + 1)]].values).T
    assert bid_prices.shape == (n_levels, len(df))
    for i in range(n_levels):
        prev_price = bid_prices[i - 1] if i > 0 else price
        bid_prices[i] = fill_zero_and_nan_with_tensor(bid_prices[i], prev_price - 0.01)
    bid_volumes = get_tensor(df[[f'BidVolume{i}' for i in range(1, n_levels + 1)]].values).T
    torch.nan_to_num_(bid_volumes, nan=0)

    return {'price': price, 'volume': volume, 'value': value,
            'ask_prices': ask_prices, 'ask_volumes': ask_volumes,
            'bid_prices': bid_prices, 'bid_volumes': bid_volumes}


def fill_zero_and_nan_with_tensor(src: TEN, dst: TEN) -> TEN:
    fill_bool = torch.logical_or(torch.isnan(src), src == 0)
    src[fill_bool] = dst[fill_bool]
    return src


'''unit tests'''


def test_get_trade_dates():
    trade_dates = get_trade_dates(beg_date='2022.09.01', end_date='2022.09.15')
    print(trade_dates)
    assert trade_dates == ['2022-09-01', '2022-09-02', '2022-09-05', '2022-09-06', '2022-09-07',
                           '2022-09-08', '2022-09-09', '2022-09-13', '2022-09-14', '2022-09-15']


def test_csv_to_tensor_dict():
    gpu_id = 0
    share_dir = './shares_data_by_day_zhongzheng500'
    share_symbol = '000525_XSHE'
    trade_date = '2022-09-01'
    n_levels = 5

    csv_path = f"{share_dir}/{share_symbol}/{trade_date}.csv"
    device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")
    tensor_dict = csv_to_tensor_dict(csv_path=csv_path, device=device, n_levels=n_levels)
    for k, v in tensor_dict.items():
        print(f"{k} {v.shape}")
    assert tensor_dict['price'].shape == (4800 + 2,)
    assert tensor_dict['volume'].shape == (4800 + 2,)
    assert tensor_dict['value'].shape == (4800 + 2,)
    assert tensor_dict['ask_prices'].shape == (n_levels, 4800 + 2)
    assert tensor_dict['ask_volumes'].shape == (n_levels, 4800 + 2)
    assert tensor_dict['bid_prices'].shape == (n_levels, 4800 + 2)
    assert tensor_dict['bid_volumes'].shape == (n_levels, 4800 + 2)


def test_get_share_dicts_by_day():
    gpu_id = 0
    share_dir = 'share_daily_zhongzheng500'
    share_symbol = '000525_XSHE'
    n_levels = 5
    n_days = 5
    device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")

    tensor_dicts = get_share_dicts_by_day(share_dir=share_dir, share_symbol=share_symbol,
                                          beg_date='2022-09-01', end_date='2022-09-30',
                                          n_levels=n_levels, device=device)
    for tensor_dict in tensor_dicts:
        trade_date = tensor_dict['trade_date']
        share_symbol = tensor_dict['share_symbol']
        print(f"trade_date {trade_date}    share_symbol {share_symbol}")
        for k, v in tensor_dict.items():
            if isinstance(v, str):
                continue
            v_info = v.shape if isinstance(v, TEN) else v
            print(f"    {k} {v_info}")

    tensor_dicts = _add_norm(tensor_dicts=tensor_dicts, n_days=n_days)
    for tensor_dict in tensor_dicts:
        trade_date = tensor_dict['trade_date']
        print(f"trade_date {trade_date}")
        for k in ('price_norm', 'volume_norm', 'value_norm'):
            avg, std = tensor_dict[k]
            print(f"    {k:16}    avg {avg:10.3e}    std {std:10.3e}")

    tensor_dicts = _add_share_style(tensor_dicts=tensor_dicts, n_days=n_days,
                                    data_dir='./share_mkt_equ_daily_zhongzheng500')
    for tensor_dict in tensor_dicts:
        trade_date = tensor_dict['trade_date']
        turnover_rate = tensor_dict['turnover_rate']
        neg_mkt_value = tensor_dict['neg_mkt_value']
        print(f"trade_date {trade_date}    turnover_rate {turnover_rate:10.3e}    neg_mkt_value {neg_mkt_value:10.3e}")

    assert isinstance(tensor_dicts, list)
    assert isinstance(tensor_dicts[0], dict)


if __name__ == '__main__':
    # test_get_trade_dates()
    # test_csv_to_tensor_dict()
    test_get_share_dicts_by_day()
