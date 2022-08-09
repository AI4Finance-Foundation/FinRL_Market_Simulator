"""
Simulated environment for trade execution

Terminology
  Time tick: The minimal time interval of our raw data = 3s
  Bar: The time interval of our simulated environment
  Horizon: The total time interval for the sequential decision problem
"""

import os
import pdb
import pickle
import pandas as pd
import numpy as np
from pathos.multiprocessing import ProcessingPool as Pool
from constants import CODE_LIST, JUNE_DATE_LIST


NUM_CORES = 40


FEATURE_SET_LOB = [
    'bidPrice1', 'bidPrice2', 'bidPrice3', 'bidPrice4', 'bidPrice5',
    'bidVolume1', 'bidVolume2', 'bidVolume3', 'bidVolume4', 'bidVolume5',  
    'askPrice1', 'askPrice2', 'askPrice3', 'askPrice4', 'askPrice5', 
    'askVolume1', 'askVolume2', 'askVolume3', 'askVolume4', 'askVolume5', 
    'high_low_price_diff', 'close_price', 'volume', 'vwap', 'time_diff'
]

FEATURE_SET_FULL = FEATURE_SET_LOB + [
    'ask_bid_spread', 'ab_volume_misbalance', 'transaction_net_volume', 
    'volatility', 'trend', 'immediate_market_order_cost_bid', 
    'VOLR', 'PCTN_1min', 'MidMove_1min', 'weighted_price', 'order_imblance', 
    'trend_strength'
]


class DefaultConfig(object):

    path_raw_data = '/data/execution_data_v2/raw'
    # path_pkl_data = '/data/execution_data/pkl'
    path_pkl_data = '/mnt/execution_data_v2/pkl'
    result_path = 'results/exp_env'

    code_list = CODE_LIST
    date_list = JUNE_DATE_LIST

    # ############################### Trade Setting Parameters ###############################
    # Planning horizon is 30mins
    simulation_planning_horizon = 30
    # Order volume = total volume / simulation_num_shares
    simulation_num_shares = 10
    # Total volume to trade w.r.t. the basis volume
    simulation_volume_ratio = 0.005
    # Encourage a uniform liquidation strategy
    simulation_linear_reg_coeff = 0.1
    # Features used for the market variable
    simulation_features = FEATURE_SET_FULL
    # Stack the features of the previous x bars
    simulation_loockback_horizon = 5
    # Whether return flattened or stacked features of the past x bars
    simulation_do_feature_flatten = True
    # A liquidation task
    simulation_direction = 'sell'
    # If the quantity is not fully filled at the last time step, 
    #   we place an MO to fully liquidate and further plus a penalty (unit: bp)
    simulation_not_filled_penalty_bp = 2.0
    # Use discrete actions (unit: relative bp)
    simulation_discrete_actions = \
        np.concatenate([[-50, -40, -30, -25, -20, -15], np.linspace(-10, 10, 21), [15, 20, 25, 30, 40, 50]])
    # Scale the price delta if we use continuous actions
    simulation_continuous_action_scale = 10
    # Use 'discrete' or 'continuous' action space?
    simulation_action_type = 'discrete'
    # ############################### END ###############################


class DataPrepare(object):
    """
    For data preparation: 
        Parse raw csv files to pickle files required by the simulated environment 
        I.e., we transform time-tick-level csv data into bar level pkl data
    """
    def __init__(self, config):

        self.config = config

        if not os.path.isdir(self.config.path_raw_data):
            self.download_raw_data()

        os.makedirs(self.config.path_pkl_data, exist_ok=True)
        file_paths = self.obtain_file_paths()

        pool = Pool(NUM_CORES)
        res = pool.map(self.process_file, file_paths)
        pd.DataFrame(res).to_csv('data_generation_report.csv')

    def download_raw_data(self):

        raise NotImplementedError

    def process_file(self, paths, debug=True):

        csv_path, pkl_path = paths

        # Step 1: Read data
        data = pd.read_csv(csv_path, index_col=0)
        csv_shape0, csv_shape1 = data.shape

        # Filter out abnormal files (e.g., the stock is not traded on this day)
        if csv_shape0 == 1:
            return dict(csv_path=csv_path, pkl_path=pkl_path, status='EMPTY')
        if data['volume'].max() <= 0:
            return dict(csv_path=csv_path, pkl_path=pkl_path, status='NO_VOL')
        if data['lastPrice'][data['lastPrice'] > 0].mean() >= 1.09 * data['prevClosePrice'].values[0]:
            return dict(csv_path=csv_path, pkl_path=pkl_path, status='LIMIT_UP')
        if data['lastPrice'][data['lastPrice'] > 0].mean() <= 0.91 * data['prevClosePrice'].values[0]:
            return dict(csv_path=csv_path, pkl_path=pkl_path, status='LIMIT_DO')

        if debug:
            print('Current process: {} {} Shape: {}'.format(csv_path, pkl_path, data.shape))
            assert csv_shape1 == 34

        # Step 2: Formatting the raw data
        trade_date = data.iloc[0]['tradeDate']
        data.index = pd.DatetimeIndex(trade_date + ' ' + data['dataTime'])
        data = data.resample('3S', closed='right', label='right').last().fillna(method='ffill')
        data['time'] = data.index

        # Calculate delta values
        data['volume_dt'] = (data['volume'] - data['volume'].shift(1)).fillna(0)
        data['value_dt'] = (data['value'] - data['value'].shift(1)).fillna(0)

        # Exclude call auction
        data = data[data['time'].between(trade_date + ' 09:30:00', trade_date + ' 14:57:00')]
        data = data[~data['time'].between(trade_date + ' 11:30:01', trade_date + ' 12:59:59')]

        # Step 3: Backtest required bar-level information
        # Convert to 1min bar
        #   1) current snapshot (5 levels of ask/bid price/volume)
        #   2) the lowest/highest ask/bid price that yields partial execution
        ask1_deal_volume_tick = ((data['value_dt'] - data['volume_dt'] * data['bidPrice1']) \
            / (data['askPrice1'] - data['bidPrice1'])).clip(upper=data['volume_dt'], lower=0)
        bid1_deal_volume_tick = ((data['volume_dt'] * data['askPrice1'] - data['value_dt']) \
            / (data['askPrice1'] - data['bidPrice1'])).clip(upper=data['volume_dt'], lower=0)

        max_last_price = data['lastPrice'].resample('T').max().reindex(data.index).fillna(method='ffill')
        min_last_price = data['lastPrice'].resample('T').min().reindex(data.index).fillna(method='ffill')

        ask1_deal_volume = ((data['askPrice1'] == max_last_price) * ask1_deal_volume_tick).resample('T').sum()
        bid1_deal_volume = ((data['bidPrice1'] == min_last_price) * bid1_deal_volume_tick).resample('T').sum()
        max_last_price = data['askPrice1'].resample('T').max()
        min_last_price = data['bidPrice1'].resample('T').min()

        # Current 5-level ask/bid price/volume (for modeling temporary market impact of MOs)
        level_infos = ['bidPrice1', 'bidVolume1', 'bidPrice2', 'bidVolume2', 'bidPrice3', 'bidVolume3', 'bidPrice4',
            'bidVolume4', 'bidPrice5', 'bidVolume5', 'askPrice1', 'askVolume1', 'askPrice2', 'askVolume2', 'askPrice3', 
            'askVolume3', 'askPrice4', 'askVolume4', 'askPrice5', 'askVolume5']
        bar_data = data[level_infos].resample('T').first()

        # Fix a common bug in data: level data is missing in the last snapshot
        bar_data.iloc[-1].replace(0.0, np.nan, inplace=True)
        bar_data.fillna(method='ffill', inplace=True)

        # Lowest ask/bid executable price and volume till the next bar (for modeling temporary market impact of LOs)
        bar_data['max_last_price'] = max_last_price
        bar_data['min_last_price'] = min_last_price
        bar_data['ask1_deal_volume'] = ask1_deal_volume
        bar_data['bid1_deal_volume'] = bid1_deal_volume

        # Step 4: Generate state features
        # Normalization constant
        raise NotImplementedError

        # Step 5: Save to pickle
        with open(pkl_path, 'wb') as f:
            pickle.dump(bar_data, f, pickle.HIGHEST_PROTOCOL)

        return dict(csv_path=csv_path, pkl_path=pkl_path, 
            csv_shape0=csv_shape0, csv_shape1=csv_shape1, 
            res_shape0=bar_data.shape[0], res_shape1=bar_data.shape[1])

    @staticmethod
    def _calculate_immediate_market_order_cost(bar_data, direction='ask'):

        # Assume the market order quantity is 1/500 of the basis volume
        remaining_quantity = (bar_data['basis_volume'] / 500).copy()
        total_fee = pd.Series(0, index=bar_data.index)
        for i in range(1, 6):
            total_fee = total_fee \
                + bar_data['{}Price{}'.format(direction, i)] \
                * np.minimum(bar_data['{}Volume{}'.format(direction, i)], remaining_quantity)
            remaining_quantity = (remaining_quantity - bar_data['{}Volume{}'.format(direction, i)]).clip(lower=0)
        if direction == 'ask':
            return total_fee / (bar_data['basis_volume'] / 500) - bar_data['askPrice1']
        elif direction == 'bid':
            return bar_data['bidPrice1'] - total_fee / (bar_data['basis_volume'] / 500)

    def obtain_file_paths(self):

        file_paths = []
        tickers = os.listdir(self.config.path_raw_data)
        for ticker in tickers:
            dates = os.listdir(os.path.join(self.config.path_raw_data, ticker))
            file_paths.extend([
                (os.path.join(self.config.path_raw_data, ticker, date), 
                 os.path.join(self.config.path_pkl_data, ticker, date.split('.')[0] + '.pkl')) for date in dates])
            os.makedirs(os.path.join(self.config.path_pkl_data, ticker), exist_ok=True)
        return file_paths


# Support the data interation in the simulated environment
class Data(object):

    price_5level_features = [
        'bidPrice1', 'bidPrice2', 'bidPrice3', 'bidPrice4', 'bidPrice5',
        'askPrice1', 'askPrice2', 'askPrice3', 'askPrice4', 'askPrice5', 
    ]
    other_price_features = [
        'high_price', 'low_price', 'open_price', 'close_price', 'vwap',
    ]
    price_delta_features = [
        'ask_bid_spread', 'trend', 'immediate_market_order_cost_ask', 
        'immediate_market_order_cost_bid', 'volatility', 'high_low_price_diff',
    ]
    volume_5level_features = [
        'bidVolume1', 'bidVolume2', 'bidVolume3', 'bidVolume4', 'bidVolume5',  
        'askVolume1', 'askVolume2', 'askVolume3', 'askVolume4', 'askVolume5', 
    ]
    other_volume_features = [
        'volume', 'ab_volume_misbalance', 'transaction_net_volume', 
        'VOLR', 'BSP',
    ]
    backtest_lo_features = [
        'max_last_price', 'min_last_price', 'ask1_deal_volume', 'bid1_deal_volume',
    ]

    def __init__(self, config):

        self.config = config
        self.data = None
        self.backtest_data = None

    def _maintain_backtest_data(self):

        self.backtest_data = \
            self.data[self.price_5level_features + self.volume_5level_features + self.backtest_lo_features].copy()
        self.backtest_data['latest_price'] = \
            (self.backtest_data['askPrice1'] + self.backtest_data['bidPrice1']) / 2

    def _normalization(self):

        # Keep normalization units
        self.basis_price = self.backtest_data.loc[self.start_index, 'latest_price']
        self.basis_volume = self.data['basis_volume'].values[0]
        # Approximation: Average price change 2% * 50 = 1.0
        self.data[self.price_5level_features] = \
            (self.data[self.price_5level_features] - self.basis_price) / self.basis_price * 50
        self.data[self.other_price_features] = \
            (self.data[self.other_price_features] - self.basis_price) / self.basis_price * 50
        self.data[self.price_delta_features] = \
            self.data[self.price_delta_features] / self.basis_price * 10

        # Such that the volumes are equally distributed in the range [-1, 1]
        self.data[self.volume_5level_features] = \
            self.data[self.volume_5level_features] / self.basis_volume * 100
        self.data[self.other_volume_features] = \
            self.data[self.other_volume_features] / self.basis_volume * 100

    def data_exists(self, code='300733.XSHE', date='2021-09-24'):

        return os.path.isfile(os.path.join(self.config.path_pkl_data, code, date + '.pkl'))

    def obtain_data(self, code='300733.XSHE', date='2021-09-24', start_index=None, do_normalization=True):

        with open(os.path.join(self.config.path_pkl_data, code, date + '.pkl'), 'rb') as f:
            self.data = pickle.load(f)
        assert self.data.shape[0] == 239, \
            'The data should be of the shape (239, 42), instead of {}'.format(self.data.shape)
        if start_index is None:
            # randomly choose a valid start_index
            start_index = self._random_valid_start_index()
            self._set_horizon(start_index)
        else:
            self._set_horizon(start_index)
            assert self._sanity_check(), "code={} date={} with start_index={} is invalid".format(code, date, start_index)
        self._maintain_backtest_data()
        if do_normalization:
            self._normalization()

    def _random_valid_start_index(self):
        cols = ['bidPrice1', 'bidVolume1', 'askPrice1', 'askVolume1']

        tmp = (self.data[cols] > 0).all(axis=1)
        tmp1 = tmp.rolling(self.config.simulation_loockback_horizon).apply(lambda x: x.all())
        tmp2 = tmp[::-1].rolling(self.config.simulation_planning_horizon + 1).apply(lambda x: x.all())[::-1]
        available_indx = tmp1.loc[(tmp1 > 0) & (tmp2  > 0)].index.tolist()
        assert len(available_indx) > 0, "The data is invalid"
        return np.random.choice(available_indx)

    def random_start_index(self):
        """deprecated"""
        return np.random.randint(self.config.simulation_loockback_horizon - 1, 239 - self.config.simulation_planning_horizon)

    def pick_horizon(self):
        """deprecated"""
        self.start_index = np.random.randint(self.config.simulation_loockback_horizon - 1, 239 - self.config.simulation_planning_horizon)
        self.current_index = self.start_index
        self.end_index = self.start_index + self.config.simulation_planning_horizon

    def _set_horizon(self, start_index):
        self.start_index = start_index
        self.current_index = self.start_index
        self.end_index = self.start_index + self.config.simulation_planning_horizon

    def obtain_features(self, do_flatten=True):
        features = self.data.loc[self.current_index - self.config.simulation_loockback_horizon + 1: self.current_index, 
            self.config.simulation_features][::-1].values
        if do_flatten:
            return features.flatten()
        else:
            return features

    def obtain_future_features(self, features):
        return self.data.loc[self.current_index:self.end_index, features]

    def obtain_level(self, name, level=''):
        return self.backtest_data.loc[self.current_index, '{}{}'.format(name, level)]

    def step(self):
        self.current_index += 1

    def _sanity_check(self):
        """ When the price reaches daily limit, the price and volume"""
        cols = ['bidPrice1', 'bidVolume1', 'askPrice1', 'askVolume1']
        if (self.data.loc[self.start_index:self.end_index, cols] == 0).any(axis=None):
            return False
        else:
            return True


class BaseWrapper(object):
    def __init__(self, env):
        self.env = env 

    def reset(self, code=None, date=None, start_index=None):
        return self.env.reset(code, date, start_index)

    def step(self, action):
        return self.env.step(action)

    @property
    def quantity(self):
        return self.env.quantity

    @property
    def total_quantity(self):
        return self.env.total_quantity

    @property
    def cash(self):
        return self.env.cash

    @property
    def config(self):
        return self.env.config

    @property
    def data(self):
        return self.env.data

    @property
    def observation_dim(self):
        return self.env.observation_dim

    def get_metric(self, mtype='IS'):
        return self.env.get_metric(mtype)

    def get_future(self, features, padding=None):
        return self.env.get_future(features, padding=padding)


class DiscreteActionBaseWrapper(BaseWrapper):
    def __init__(self, env):
        super(DiscreteActionBaseWrapper, self).__init__(env)

    @property
    def action_sample_func(self):
        return lambda: np.random.randint(len(self.discrete_actions))

    @property
    def action_dim(self):
        return len(self.discrete_actions)
        

class DiscretePriceQuantityWrapper(DiscreteActionBaseWrapper):
    def __init__(self, env):
        super(DiscretePriceQuantityWrapper, self).__init__(env)
        self.discrete_actions = self.config.simulation_discrete_actions
        self.simulation_discrete_quantities = self.config.simulation_discrete_quantities
        self.base_quantity_ratio = self.config.simulation_volume_ratio \
            / self.config.simulation_num_shares / self.simulation_discrete_quantities

    def step(self, action):
        price, quantity = self.discrete_actions[action]
        price = np.round((1 + price / 10000) * self.data.obtain_level('askPrice', 1) * 100) / 100
        quantity = self.data.basis_volume * self.base_quantity_ratio * quantity
        return self.env.step(dict(price=price, quantity=quantity))


class DiscreteQuantityNingWrapper(DiscreteActionBaseWrapper):
    """
    Follows [Ning et al 2020]
    Divide the remaining quantity into several parts and trade using MO
    """
    def __init__(self, env):
        super(DiscreteQuantityNingWrapper, self).__init__(env)
        self.discrete_actions = self.config.simulation_discrete_actions

    def step(self, action):
        quantity = self.discrete_actions[action]
        # This ensures that this can be an MO
        price = -50 
        price = np.round((1 + price / 10000) * self.data.obtain_level('askPrice', 1) * 100) / 100
        quantity = self.discrete_actions[action] / (len(self.discrete_actions) - 1) * self.quantity
        return self.env.step(dict(price=price, quantity=quantity))


class DiscreteQuantityWrapper(DiscreteActionBaseWrapper):
    """
    Specify the quantity and trade using MO
    """
    def __init__(self, env):
        super(DiscreteQuantityWrapper, self).__init__(env)
        self.discrete_actions = self.config.simulation_discrete_actions

    def step(self, action):
        quantity = self.discrete_actions[action]
        # This ensures that this can be an MO
        price = -50 
        price = np.round((1 + price / 10000) * self.data.obtain_level('askPrice', 1) * 100) / 100
        quantity = self.discrete_actions[action] / (len(self.discrete_actions) - 1) * self.total_quantity
        return self.env.step(dict(price=price, quantity=quantity))


class DiscretePriceWrapper(DiscreteActionBaseWrapper):
    """
    The quantity is fixed and equals to total_quantity 
    """
    def __init__(self, env):
        super(DiscretePriceWrapper, self).__init__(env)
        self.discrete_actions = self.config.simulation_discrete_actions
        self.num_shares = self.config.simulation_num_shares

    def step(self, action):
        price = self.discrete_actions[action]
        price = np.round((1 + price / 10000) * self.data.obtain_level('askPrice', 1) * 100) / 100
        quantity = self.total_quantity / self.num_shares
        return self.env.step(dict(price=price, quantity=quantity))


class ContinuousActionWrapper(BaseWrapper):
    def __init__(self, env):
        super(ContinuousActionWrapper, self).__init__(env)
        self.fixed_quantity_ratio = self.config.simulation_volume_ratio / self.config.simulation_num_shares
        self.num_shares = self.config.simulation_num_shares

    def step(self, action):
        price = self.continuous_action_scale * action
        price = np.round((1 + price / 10000) * self.data.obtain_level('askPrice', 1) * 100) / 100
        quantity = self.total_quantity / self.num_shares
        return self.env.step(dict(price=price, quantity=quantity))


def make_env(config):
    if config.simulation_action_type == 'discrete_p':
        return DiscretePriceWrapper(ExecutionEnv(config))
    elif config.simulation_action_type == 'continuous':
        return ContinuousActionWrapper(ExecutionEnv(config))
    elif config.simulation_action_type == 'discrete_pq':
        return DiscretePriceQuantityWrapper(ExecutionEnv(config))
    elif config.simulation_action_type == 'discrete_q_ning':
        return DiscreteQuantityNingWrapper(ExecutionEnv(config))
    elif config.simulation_action_type == 'discrete_q':
        return DiscreteQuantityWrapper(ExecutionEnv(config))


class ExecutionEnv(object):
    """
    Simulated environment for trade execution
      Feature 1: There is no model misspecification error since the simulator is based on historical data.
      Featrue 2: We can model temporary market impact for MO and LO. 
          For MO, we assume that the order book is resilient between bars
          For LO, we assume that 1) full execution when the price passes through; 
              2) partial execution when the price reaches; 3) no execution otherwise
    """
    def __init__(self, config):
        self.config = config
        self.current_code = None 
        self.current_date = None 
        self.data = Data(config)
        self.cash = 0
        self.total_quantity = 0
        self.quantity = 0
        self.valid_code_date_list = self.get_valid_code_date_list()

    def get_valid_code_date_list(self):
        code_date_list = []
        for code in self.config.code_list:
            for date in self.config.date_list:
                if self.data.data_exists(code, date):
                    code_date_list.append((code, date))
        return code_date_list

    def reset(self, code=None, date=None, start_index=None):

        count = 0

        while True:
            if code is None and date is None:
                # Uniformly randomly select a code and date
                ind = np.random.choice(len(self.valid_code_date_list))
                self.current_code, self.current_date = self.valid_code_date_list[ind]
            else:
                self.current_code, self.current_date = code, date

            try:
                self.data.obtain_data(self.current_code, self.current_date, start_index)
                break
            except AssertionError as e:
                count += 1
                print('Invalid: code={} date={}'.format(self.current_code, self.current_date))
                if count > 100:
                    raise ValueError("code={} date={} is invalid".format(code, date))
            except Exception as e:
                raise e

        self.cash = 0
        self.total_quantity = self.config.simulation_volume_ratio * self.data.basis_volume
        self.quantity = self.total_quantity
        self.latest_price = self.data.obtain_level('latest_price')

        # Notice that we do not use the first time step of the day (with zero volum)
        market_state = self.data.obtain_features(do_flatten=self.config.simulation_do_feature_flatten)
        private_state = self._generate_private_state()

        return market_state, private_state

    def _generate_private_state(self):
        elapsed_time = (self.data.current_index - self.data.start_index) / self.config.simulation_planning_horizon
        remaining_quantity = self.quantity / self.total_quantity
        return np.array([elapsed_time, remaining_quantity])

    def get_future(self, features, padding=None):
        future = self.data.obtain_future_features(features)
        if padding is None:
            return future
        else:
            padding_width = padding - future.shape[0]
            future = np.pad(future, ((0, padding_width), (0, 0)), 'edge')
            return future

    def step(self, action=dict(price=20.71, quantity=300)):
        if self.config.simulation_direction == 'sell':
            return self._step_sell(action)
        else:
            raise NotImplementedError

    def _step_sell(self, action=dict(price=20.71, quantity=300)):
        """
        We only consider limit orders.
        If the price is no better than the market order, 
            it will be transformed to market order automatically.
        """

        info = dict(
            code=self.current_code, 
            date=self.current_date, 
            start_index=self.data.start_index, 
            end_index=self.data.end_index,
            current_index=self.data.current_index
        )
        order_quantity = action['quantity']
        pre_quantity = self.quantity
        pre_cash = self.cash
        price_penalty = 0.0

        done = (self.data.current_index + 1 >= self.data.end_index)
        if done:
            action['price'] = 0.0
            action['quantity'] = float('inf')
            price_penalty = self.config.simulation_not_filled_penalty_bp / 10000 * self.data.basis_price

        # Step 1: If can be executed immediately
        for level in range(1, 6):
            if action['quantity'] > 0 and action['price'] <= self.data.obtain_level('bidPrice', level):
                executed_volume = min(self.data.obtain_level('bidVolume', level), action['quantity'], self.quantity)
                self.cash += executed_volume * (self.data.obtain_level('bidPrice', level) - price_penalty)
                self.quantity -= executed_volume
                action['quantity'] -= executed_volume

        # Liquidate all the remaining inventory on the last step
        if done:
            executed_volume = self.quantity
            self.cash += executed_volume * (self.data.obtain_level('bidPrice', 5) - price_penalty)
            self.quantity = 0
            action['quantity'] = 0

        # Step 2: If can be executed until the next bar
        if action['price'] < self.data.obtain_level('max_last_price'):
            executed_volume = min(self.quantity, action['quantity'])
            self.cash += executed_volume * action['price']
            self.quantity -= executed_volume
            action['quantity'] -= executed_volume
        elif action['price'] == self.data.obtain_level('max_last_price'):
            executed_volume = min(self.quantity, action['quantity'], self.data.obtain_level('ask1_deal_volume'))
            self.cash += executed_volume * action['price']
            self.quantity -= executed_volume
            action['quantity'] -= executed_volume

        if action['quantity'] == order_quantity:
            info['status'] = 'NOT_FILLED'
        elif action['quantity'] == 0:
            info['status'] = 'FILLED'
        else:
            info['status'] = 'PARTIAL_FILLED'

        # Step 3: Reward/Done calculation
        if not done:
            self.data.step()

        reward = self._calculate_reward_v1(pre_cash)

        market_state = self.data.obtain_features(do_flatten=self.config.simulation_do_feature_flatten)
        private_state = self._generate_private_state()

        return market_state, private_state, reward, done, info

    def _calculate_reward_v1(self, pre_cash):
        _recommand_quantity = self.total_quantity * (self.data.end_index - self.data.current_index) \
            / self.config.simulation_planning_horizon
        basic_reward = (self.cash - pre_cash) / self.data.basis_price / \
            (self.data.basis_volume * self.config.simulation_volume_ratio / self.config.simulation_planning_horizon)
        linear_reg = abs(self.quantity - _recommand_quantity) / \
            (self.data.basis_volume * self.config.simulation_volume_ratio / self.config.simulation_planning_horizon)
        return basic_reward - self.config.simulation_linear_reg_coeff * linear_reg

    def _calculate_reward_v2(self, price_diff):
        """ problematic """
        _recommand_quantity = self.total_quantity * (self.data.end_index - self.data.current_index) \
            / self.config.simulation_planning_horizon
        basic_reward = price_diff * self.quantity / self.data.basis_volume
        linear_reg = self.data.basis_price * ((self.quantity - _recommand_quantity) ** 2) \
            / (self.data.basis_volume ** 2)
        return basic_reward - self.config.simulation_linear_reg_coeff * linear_reg

    @property
    def observation_dim(self):
        return len(self.config.simulation_features) * self.config.simulation_loockback_horizon

    def get_metric(self, mtype='IS'):
        # IS: implementation shortfall
        if mtype == 'IS': 
            return self.data.basis_price * (self.total_quantity - self.quantity) - self.cash
        # BP: bp over mid price TWAP
        if mtype == 'BP':
            if self.total_quantity == self.quantity:
                return 0
            avg_price = self.cash / (self.total_quantity - self.quantity)
            TWAP_mid = self.data.backtest_data.loc[self.data.start_index:self.data.end_index, 'latest_price'].mean()
            bp = (avg_price - TWAP_mid) / self.data.basis_price * 10000
            return bp


def run_data_prepare():
    
    config = DefaultConfig()
    data_prepare = DataPrepare(config)

def run_env_test():

    config = DefaultConfig()
    env = make_env(config)

    market_state, private_state = env.reset()
    print('market_state = {}'.format(market_state))
    print('private_state = {}'.format(private_state))
    print('snapshot = ')
    print(env.data.backtest_data.loc[env.data.current_index])

    market_state, private_state, reward, done, info = env.step(0.5)
    print('market_state = {}'.format(market_state))
    print('private_state = {}'.format(private_state))
    print('reward = {}'.format(reward))
    print('done = {}'.format(done))
    print('info = {}'.format(info)) 
    print('snapshot = ')
    print(env.data.backtest_data.loc[env.data.current_index])

    market_state, private_state, reward, done, info = env.step(-0.5)
    print('market_state = {}'.format(market_state))
    print('private_state = {}'.format(private_state))
    print('reward = {}'.format(reward))
    print('done = {}'.format(done))
    print('info = {}'.format(info))

if __name__ == '__main__':

    run_data_prepare()
    run_env_test()