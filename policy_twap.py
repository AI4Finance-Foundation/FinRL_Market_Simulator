"""
TWAP strategy
"""

import torch
import torch.nn as nn
import torch.optim as opt
from torch import Tensor
from torch.autograd import Variable
import torch.nn.functional as F

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from constants import CODE_LIST, JUNE_DATE_LIST, VALIDATION_DATE_LIST, VALIDATION_CODE_LIST
from env import make_env

from pathos.multiprocessing import ProcessingPool as Pool
from sklearn.preprocessing import StandardScaler
from scipy.special import softmax, expit
from collections import deque
from tqdm import trange
import pandas as pd
import numpy as np
import itertools
import pdb
import os 


class DefaultConfig(object):

    path_raw_data = '/mnt/execution_data_v2/raw'
    # path_pkl_data = '/data/execution_data/pkl'
    path_pkl_data = '/mnt/execution_data_v2/pkl'
    result_path = 'results/exp34'

    code_list = CODE_LIST
    date_list = JUNE_DATE_LIST
    code_list_validation = VALIDATION_CODE_LIST
    date_list_validation = VALIDATION_DATE_LIST

    # Selected features
    simulation_features = [
        'bidPrice1', 'bidPrice2', 'bidPrice3', 'bidPrice4', 'bidPrice5',
        'bidVolume1', 'bidVolume2', 'bidVolume3', 'bidVolume4', 'bidVolume5',  
        'askPrice1', 'askPrice2', 'askPrice3', 'askPrice4', 'askPrice5', 
        'askVolume1', 'askVolume2', 'askVolume3', 'askVolume4', 'askVolume5', 
        'high_low_price_diff', 'close_price', 'volume', 'vwap', 'time_diff',
        'ask_bid_spread', 'ab_volume_misbalance', 'transaction_net_volume', 'volatility',  
        'trend', 'immediate_market_order_cost_bid', 
    ]

    # Stack the features of the previous x bars
    simulation_loockback_horizon = 5

    # Whether return flattened or stacked features of the past x bars
    simulation_do_feature_flatten = True

    # ############################### Trade Setting 1 Parameters ###############################
    # # Planning horizon is 30mins
    # simulation_planning_horizon = 30
    # # Total volume to trade w.r.t. the basis volume
    # simulation_volume_ratio = 0.005
    # # Type of action space
    # simulation_action_type = 'discrete_p'
    # # Order volume = total volume / simulation_num_shares
    # simulation_num_shares = 10
    # # Use discrete actions
    # simulation_discrete_actions = np.linspace(-30, 30, 61)
    # ############################### END ######################################################

    # ############################### Trade Setting 2 Parameters ###############################
    # Planning horizon is 30mins
    simulation_planning_horizon = 30
    # Total volume to trade w.r.t. the basis volume
    simulation_volume_ratio = 0.005
    # Type of action space
    simulation_action_type = 'discrete_q'
    # Use discrete actions
    simulation_discrete_actions = np.arange(31)
    # ############################### END ######################################################

    simulation_direction = 'sell'
    # Quadratic penalty to minimize the impact of permanent market impact
    # Penalty = coeff * basis_price / basis_volume
    # Encourage a uniform liquidation strategy
    simulation_linear_reg_coeff = 0.1
    # If the quantity is not fully filled at the last time step, we place an MO to liquidate and further plus a penalty
    simulation_not_filled_penalty_bp = 2.0
    # Scale the price delta if we use continuous actions
    simulation_continuous_action_scale = 10
    # Scale the reward to approx. unit range
    simulation_reward_scale = 1000


class TWAP_Agent(object):
    def __init__(self):
        super(TWAP_Agent, self).__init__()

    def act(self, market_state, private_state):
        elapsed_time = private_state[0]
        executed_quantity = 1 - private_state[1]
        if elapsed_time >= executed_quantity:
            return 0
        else:
            return 60


class TWAP_Agent2(object):
    def __init__(self):
        super(TWAP_Agent2, self).__init__()

    def act(self, market_state, private_state):
        return 1


class Evaluation(object):
    def __init__(self, config):
        super(Evaluation, self).__init__()
        self.config = config
        self.env = make_env(config)

    def evaluate(self, agent):

        def run(dumb):
            bps = []
            rews = []
            for code in self.config.code_list_validation:
                for date in self.config.date_list_validation:
                    record = self.evaluate_single(agent, code=code, date=date)
                    bps.append(record['BP'].values[-1])
                    rews.append(record['reward'].sum())
            return np.mean(bps), np.mean(rews)

        pool = Pool(80)
        record = pool.map(run, list(range(1000)))
        bp_list = [item[0] for item in record]
        rew_list = [item[1] for item in record]

        return dict(
            BP_avg=np.mean(bp_list),
            reward_avg=np.mean(rew_list),
            BP_std=np.std(bp_list),
            reward_std=np.std(rew_list)
        )

    def evaluate_detail_batch(self, agent, iteration=1,
        code='000504.XSHE', 
        date_list=['2021-06-01', '2021-06-03', '2021-06-04', '2021-07-02', '2021-07-05', '2021-07-06']):

        path = os.path.join(self.config.result_path, 'evaluation', 'it{:08d}'.format(iteration))
        os.makedirs(path, exist_ok=True)

        record = []
        for date in date_list:
            for i in range(5):
                res = self.evaluate_single(agent, code=code, date=date)
                record.append(res)
                Figure().plot_policy(df=res, filename=os.path.join(path, 'fig_{}_{}_{}.png'.format(code, date, i)))

        pd.concat(record).to_csv(os.path.join(path, 'detail_{}.csv'.format(code)))

    def evaluate_single(self, agent, code='600519.XSHG', date='2021-06-01'):
        record = []
        sm, sp = self.env.reset(code, date)
        done = False 
        step = 0
        action = None 
        info = dict(status=None)

        while not done:
            action = agent.act(sm, sp)
            nsm, nsp, reward, done, info = self.env.step(action)

            record.append(dict(
                code=code,
                date=date,
                step=step,
                quantity=self.env.quantity,
                action=action,
                ask_price=self.env.data.obtain_level('askPrice', 1),
                bid_price=self.env.data.obtain_level('bidPrice', 1),
                order_price=np.round((1 + self.config.simulation_discrete_actions[action] / 10000) \
                    * self.env.data.obtain_level('askPrice', 1) * 100) / 100 if action is not None else None,
                reward=reward,
                cash=self.env.cash,
                BP=self.env.get_metric('BP'),
                IS=self.env.get_metric('IS'),
                status=info['status'],
                index=self.env.data.current_index
            ))
            step += 1
            sm, sp = nsm, nsp

        return pd.DataFrame(record)


class Figure(object):
    def __init__(self):
        pass

    @staticmethod
    def plot_policy(df, filename):
        fig, ax1 = plt.subplots(figsize=(15, 6))
        ax2 = ax1.twinx()
        ax1.plot(df['index'], df['ask_price'], label='ask_price')
        ax1.plot(df['index'], df['bid_price'], label='bid_price')
        ax1.plot(df['index'], df['order_price'], label='order_price')
        ax1.legend(loc='lower left')
        ax2.plot(df['index'], df['quantity'], 'k*', label='inventory')
        ax1.set_title('{} {} BP={:.4f}'.format(df['code'].values[-1], df['date'].values[-1], df['BP'].values[-1]))
        ax2.legend(loc='upper right')
        plt.savefig(filename, bbox_inches='tight')
        plt.close('all')

    @staticmethod
    def plot_training_process_basic(df, filename):
        while df.shape[0] > 1500:
            df = df[::2]
        fig, ax1 = plt.subplots(figsize=(15, 6))
        ax2 = ax1.twinx()
        ax1.plot(df.index.values, df['reward'], 'C0', label='reward')
        ax1.legend(loc='lower left')
        ax2.plot(df.index.values, df['BP'], 'C1', label='BP')
        ax2.legend(loc='upper right')
        top_size = df.shape[0] // 10
        mean_bp_first = np.mean(df['BP'].values[:top_size])
        mean_bp_last = np.mean(df['BP'].values[-top_size:])
        mean_rew_first = np.mean(df['reward'].values[:top_size])
        mean_rew_last = np.mean(df['reward'].values[-top_size:])
        ax2.set_title('BP {:.4f}->{:.4f} reward {:.4f}->{:.4f}'.format(mean_bp_first, mean_bp_last, mean_rew_first, mean_rew_last))

        if 'loss' in df.columns:
            ax3 = ax1.twinx()
            p3, = ax3.plot(df.index.values, df['loss'], 'C2')
            ax3.yaxis.label.set_color('C2')

        plt.savefig(filename, bbox_inches='tight')
        plt.close('all')

if __name__ == '__main__':

    for i, lin_reg in enumerate([1.0, 0.1, 0.01]):
        config = DefaultConfig()
        config.simulation_linear_reg_coeff = lin_reg
        evaluation = Evaluation(config)

        agent = TWAP_Agent2()

        result = evaluation.evaluate(agent)
        print('Lin_reg={:.1E} BP={:.4f}({:.4f}) reward={:.4f}({:.4f})'\
            .format(lin_reg, result['BP_avg'], result['BP_std'], result['reward_avg'], result['reward_std']))
        evaluation.evaluate_detail_batch(agent, iteration=i+20)
        