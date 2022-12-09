import os

import torch
import pandas


def get_trade_dates(beg_date: str = '2022.09.01', end_date: str = '2022.09.15') -> [str]:
    from ideadata.stock.trade_calendar import TradeCalendar
    cal_df = TradeCalendar().get_trade_cal(beg_date, end_date)
    cal_df = cal_df[cal_df.is_open == 1]

    trade_dates = [item.date for item in cal_df.itertuples()]
    return trade_dates


def get_share_dicts_by_day(share_dir='./shares_data_by_day', share_name='000520_XSHE',
                           beg_date='2022-09-01', end_date='2022-09-30',
                           n_levels=5, n_days=5, device=None):
    if device is None:
        gpu_id = 0
        device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")

    '''convert csv to tensor by day'''
    if beg_date is None:
        trade_dates = [csv_name[:-4] for csv_name in sorted(os.listdir(share_dir))
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
        csv_path = f"{share_dir}/{trade_date}.csv"
        tensor_dict = csv_to_tensor_dict(csv_path=csv_path, device=device, n_levels=n_levels)
        tensor_dict['trade_date'] = trade_date
        tensor_dict['share_name'] = share_name
        tensor_dicts.append(tensor_dict)

    # '''get avg and the vam(the mean of var)'''
    # price_avgs = torch.stack([d['price'].mean(dim=0) for d in tensor_dicts])
    # price_vams = torch.stack([(d['price'] ** 2).mean(dim=0, keepdim=True) for d in tensor_dicts])
    # volume_avgs = torch.stack([d['volume'].mean(dim=0) for d in tensor_dicts])
    # volume_vams = torch.stack([(d['volume'] ** 2).mean(dim=0, keepdim=True) for d in tensor_dicts])
    # value_avgs = torch.stack([d['value'].mean(dim=0) for d in tensor_dicts])
    # value_vams = torch.stack([(d['value'] ** 2).mean(dim=0, keepdim=True) for d in tensor_dicts])
    #
    # '''save avg and std for normalization using previous day data'''
    #
    # def get_std(_vam, _avg):
    #     return torch.sqrt(_vam - _avg ** 2)
    #
    # for i_day in range(len(tensor_dicts)):
    #     if i_day <= n_days:
    #         i, j = 0, n_days
    #     else:
    #         i, j = i_day - n_days, i_day
    #     tensor_dict = tensor_dicts[i_day]
    #
    #     price_avg = price_avgs[i:j].mean()
    #     price_std = get_std(price_vams[i:j].mean(), price_avg)
    #     tensor_dict['price_norm'] = price_avg, price_std
    #
    #     volume_avg = volume_avgs[i:j].mean()
    #     volume_std = get_std(volume_vams[i:j].mean(), volume_avg)
    #     tensor_dict['volume_norm'] = volume_avg, volume_std
    #
    #     value_avg = value_avgs[i:j].mean()
    #     value_std = get_std(value_vams[i:j].mean(), value_avg)
    #     tensor_dict['value_norm'] = value_avg, value_std
    return tensor_dicts


def csv_to_tensor_dict(csv_path: str, device, n_levels: int = 5):
    df = pandas.read_csv(csv_path)

    def get_tensor(ary):
        return torch.tensor(ary, dtype=torch.float32, device=device)

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


def fill_zero_and_nan_with_tensor(src, dst):
    fill_bool = torch.logical_or(torch.isnan(src), src == 0)
    src[fill_bool] = dst[fill_bool]
    return src


if __name__ == '__main__':
    get_share_dicts_by_day()
