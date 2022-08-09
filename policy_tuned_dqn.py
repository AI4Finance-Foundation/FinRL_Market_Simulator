"""
Tuned DQN algorithm for optimized trade execution
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

    path_raw_data = '/data/execution_data_v2/raw'
    # path_pkl_data = '/data/execution_data/pkl'
    path_pkl_data = '/mnt/execution_data_v2/pkl'
    # path_pkl_data = os.path.expanduser('~/execution_data/pkl')
    result_path = 'results/exp36'

    code_list = CODE_LIST
    date_list = JUNE_DATE_LIST
    code_list_validation = VALIDATION_CODE_LIST
    date_list_validation = VALIDATION_DATE_LIST

    agent_scale = 100000
    agent_batch_size = 128
    agent_learn_start = 1000
    agent_gamma = 0.998
    agent_epsilon = 0.7
    agent_total_steps = 20 * agent_scale
    agent_buffer_size = agent_scale
    agent_network_update_freq = 4
    # Smooth L1 loss (SL1) or mean squared error (MSE)
    agent_loss_type = 'SL1'
    agent_lr_decay_freq = 2000
    agent_target_update_freq = 2000
    agent_eval_freq = 2000
    # Becomes 0.01 upon 70% of the training
    agent_epsilon_decay = np.exp(np.log(0.01) / (agent_scale * 0.5))
    agent_plot_freq = 20000
    agent_device = 'cuda'

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

    # ############################### Trade Setting Parameters ###############################
    # Planning horizon is 30mins
    simulation_planning_horizon = 30
    # Total volume to trade w.r.t. the basis volume
    simulation_volume_ratio = 0.005
    # Order volume = total volume / simulation_num_shares
    simulation_num_shares = 10
    # Maximum quantity is total_quantity / simulation_num_shares; further devide this into 3 levels
    simulation_discrete_quantities = 3
    # Choose the wrapper
    simulation_action_type = 'discrete_pq'
    # Discrete action space
    simulation_discrete_actions = \
        list(itertools.product(
            np.concatenate([[-50, -40, -30, -25, -20, -15], np.linspace(-10, 10, 21), [15, 20, 25, 30, 40, 50]]),
            np.arange(simulation_discrete_quantities) + 1
        ))
    # ############################### END ###############################

    # ############################### Test Parameters ###############################
    # Encourage a uniform liquidation strategy
    simulation_linear_reg_coeff = [0.1]
    agent_learning_rate = [2e-5, 1e-5, 5e-6]
    agent_network_structrue = 'MLPNetwork_complex,MLPNetwork_Xcomplex'
    # ############################### END ###############################

    # Stack the features of the previous x bars
    simulation_loockback_horizon = 5
    # Whether return flattened or stacked features of the past x bars
    simulation_do_feature_flatten = True
    simulation_direction = 'sell'
    # If the quantity is not fully filled at the last time step, we place an MO to liquidate and further plus a penalty
    simulation_not_filled_penalty_bp = 2.0
    # Scale the price delta if we use continuous actions
    # simulation_continuous_action_scale = 10


# The Q network
class MLPNetwork(nn.Module):
    def __init__(self, dim_input1, dim_input2, dim_output, hidden=128):
        super(MLPNetwork, self).__init__()
        
        self.dim_input1 = dim_input1
        self.dim_input2 = dim_input2
        self.dim_output = dim_output
        
        self.fc1 = nn.Linear(dim_input1, 2 * hidden)
        self.fc2 = nn.Linear(2 * hidden, hidden)
        self.fc3 = nn.Linear(dim_input2, hidden)
        self.fc4 = nn.Linear(2 * hidden, dim_output)
        
    def forward(self, market_states, private_states):
        x = F.relu(self.fc1(market_states))
        x = F.relu(self.fc2(x))
        y = F.relu(self.fc3(private_states))
        z = torch.cat((x, y), 1)
        z = self.fc4(z)
        return z

    def act(self, market_state, private_state, device='cuda'):
        market_state = Tensor(market_state).unsqueeze(0).to(device=device)
        private_state = Tensor(private_state).unsqueeze(0).to(device=device)
        return int(self.forward(market_state, private_state).argmax(1)[0])
        
    def act_egreedy(self, market_state, private_state, e=0.7, device='cuda'):
        return self.act(market_state, private_state, device='cuda') if np.random.rand() > e \
            else np.random.randint(self.dim_output)


# The Q network - more parameters
class MLPNetwork_complex(nn.Module):
    def __init__(self, dim_input1, dim_input2, dim_output, hidden=256):
        super(MLPNetwork_complex, self).__init__()
        
        self.dim_input1 = dim_input1
        self.dim_input2 = dim_input2
        self.dim_output = dim_output
        
        self.fc1 = nn.Linear(dim_input1, 2 * hidden)
        self.fc2 = nn.Linear(2 * hidden, hidden)
        self.fc3 = nn.Linear(dim_input2, hidden)
        self.fc4 = nn.Linear(2 * hidden, hidden)
        self.fc5 = nn.Linear(hidden, dim_output)
        
    def forward(self, market_states, private_states):
        x = F.relu(self.fc1(market_states))
        x = F.relu(self.fc2(x))
        y = F.relu(self.fc3(private_states))
        z = torch.cat((x, y), 1)
        z = F.relu(self.fc4(z))
        z = self.fc5(z)
        return z

    def act(self, market_state, private_state, device='cuda'):
        market_state = Tensor(market_state).unsqueeze(0).to(device=device)
        private_state = Tensor(private_state).unsqueeze(0).to(device=device)
        return int(self.forward(market_state, private_state).argmax(1)[0])
        
    def act_egreedy(self, market_state, private_state, e=0.7, device='cuda'):
        return self.act(market_state, private_state, device='cuda') if np.random.rand() > e \
            else np.random.randint(self.dim_output)


# The Q network - more more parameters
class MLPNetwork_Xcomplex(nn.Module):
    def __init__(self, dim_input1, dim_input2, dim_output, hidden=512):
        super(MLPNetwork_Xcomplex, self).__init__()
        
        self.dim_input1 = dim_input1
        self.dim_input2 = dim_input2
        self.dim_output = dim_output
        
        self.fc1 = nn.Linear(dim_input1, 2 * hidden)
        self.fc2 = nn.Linear(2 * hidden, hidden)
        self.fc3 = nn.Linear(dim_input2, hidden)
        self.fc4 = nn.Linear(2 * hidden, hidden)
        self.fc5 = nn.Linear(hidden, hidden)
        self.fc6 = nn.Linear(hidden, dim_output)
        
    def forward(self, market_states, private_states):
        x = F.relu(self.fc1(market_states))
        x = F.relu(self.fc2(x))
        y = F.relu(self.fc3(private_states))
        z = torch.cat((x, y), 1)
        z = F.relu(self.fc4(z))
        z = F.relu(self.fc5(z))
        z = self.fc6(z)
        return z

    def act(self, market_state, private_state, device='cuda'):
        market_state = Tensor(market_state).unsqueeze(0).to(device=device)
        private_state = Tensor(private_state).unsqueeze(0).to(device=device)
        return int(self.forward(market_state, private_state).argmax(1)[0])
        
    def act_egreedy(self, market_state, private_state, e=0.7, device='cuda'):
        return self.act(market_state, private_state, device='cuda') if np.random.rand() > e \
            else np.random.randint(self.dim_output)


# The Q network - more parameters + positional encoding
class MLPNetwork_complex_posenc(nn.Module):
    def __init__(self, dim_input1, dim_input2, dim_output, hidden=256):
        super(MLPNetwork_complex_posenc, self).__init__()
        
        self.dim_input1 = dim_input1
        self.dim_input2 = dim_input2
        self.dim_output = dim_output
        self.hidden = hidden
        
        self.fc1 = nn.Linear(dim_input1, 2 * hidden)
        self.fc2 = nn.Linear(2 * hidden, hidden)
        self.fc4 = nn.Linear(2 * hidden, hidden)
        self.fc5 = nn.Linear(hidden, dim_output)
        
    def forward(self, market_states, private_states):

        y = torch.einsum('bi, j->bij', private_states, torch.arange(self.hidden // self.dim_input2, device=private_states.device))
        y = y.view(-1, self.hidden)
        y = torch.sin(y * 12345).detach()

        x = F.relu(self.fc1(market_states))
        x = F.relu(self.fc2(x))
        z = torch.cat((x, y), 1)
        z = F.relu(self.fc4(z))
        z = self.fc5(z)
        return z

    def act(self, market_state, private_state, device='cuda'):
        market_state = Tensor(market_state).unsqueeze(0).to(device=device)
        private_state = Tensor(private_state).unsqueeze(0).to(device=device)
        return int(self.forward(market_state, private_state).argmax(1)[0])
        
    def act_egreedy(self, market_state, private_state, e=0.7, device='cuda'):
        return self.act(market_state, private_state, device='cuda') if np.random.rand() > e \
            else np.random.randint(self.dim_output)


class ReplayBuffer(object):
    """docstring for ReplayBuffer"""
    def __init__(self, maxlen):
        super(ReplayBuffer, self).__init__()
        self.maxlen = maxlen
        self.data = deque(maxlen=maxlen)
        
    def push(self, *args):
        self.data.append(args)

    def sample(self, batch_size):
        inds = np.random.choice(len(self.data), batch_size, replace=False)
        return zip(*[self.data[i] for i in inds])

    def sample_all(self):
        return zip(*list(self.data))

    def update_all(self, new_data, ind):
        for i in range(len(self.data)):
            tup = list(self.data[i])
            tup[ind] = new_data[i, :]
            self.data[i] = tuple(tup)


class Agent(object):
    def __init__(self, config):
        super(Agent, self).__init__()

        self._set_seed()

        self.config = config
        self.env = make_env(config)
        self.dim_input1 = self.env.observation_dim       # dimension of market states
        self.dim_input2 = 2                              # dimension of private states
        self.dim_output = self.env.action_dim

        network = config.agent_network_structrue
        self.network = network(self.dim_input1, self.dim_input2, self.dim_output).to(device=self.config.agent_device)
        self.network_target = network(self.dim_input1, self.dim_input2, self.dim_output).to(device=self.config.agent_device)
        self.network_target.load_state_dict(self.network.state_dict())
        self.optimizer = opt.Adam(self.network.parameters(), lr=config.agent_learning_rate)
        self.scheduler = opt.lr_scheduler.StepLR(self.optimizer, step_size=config.agent_lr_decay_freq, gamma=0.998)
        self.buffer = ReplayBuffer(self.config.agent_buffer_size)
        self.evaluation = Evaluation(self.config)
        if config.agent_loss_type == 'MSE':
            self.loss_func = nn.MSELoss()
        elif config.agent_loss_type == 'SL1':
            self.loss_func = F.smooth_l1_loss

    def _set_seed(self, seed=None):
        if seed is None:
            seed = int.from_bytes(os.urandom(4), byteorder='little')
        else:
            seed = seed + 1234
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _filter(state):
        return np.clip(state, -3, 3)

    def _to_tensor(self, tensor, dtype=torch.float):
        return torch.tensor(tensor, dtype=dtype, device=self.config.agent_device)

    def learn(self):

        train_record = []
        eval_record = []
        reward = 0
        eplen = 0
        loss = 0
        avg_Q = 0
        epsilon = self.config.agent_epsilon
        ms_scaler = StandardScaler()

        sm, sp = self.env.reset()

        for i in trange(self.config.agent_total_steps):

            # Step 1: Execute one step and store it to the replay buffer
            if i <= self.config.agent_learn_start:
                a = self.env.action_sample_func()
            else:
                tsm = ms_scaler.transform(sm.reshape(1, -1)).flatten()
                a = self.network.act_egreedy(tsm, sp, e=epsilon, device=self.config.agent_device)

            nsm, nsp, r, done, info = self.env.step(a)

            self.buffer.push(sm, sp, a, r, nsm, nsp, done)
            reward += r
            eplen += 1
            if done:
                train_record.append(dict(
                    i=i, 
                    reward=reward,
                    eplen=eplen,
                    epsilon=epsilon,
                    lr=self.optimizer.param_groups[0]['lr'],
                    loss=float(loss),
                    avg_Q=float(avg_Q),
                    BP=self.env.get_metric('BP'),
                    IS=self.env.get_metric('IS'),
                    code=info['code'],
                    date=info['date'],
                    start_index=info['start_index']
                    ))
                reward = 0
                eplen = 0
                epsilon = max(0.01, epsilon * self.config.agent_epsilon_decay)
                sm, sp = self.env.reset()
            else:
                sm, sp = nsm, nsp

            # Step 2: Estimate variance for market states
            if i == self.config.agent_learn_start:
                market_states, _, _, _, nmarket_states, _, _ = self.buffer.sample_all()
                ms_scaler.fit(np.array(market_states))

                # Since we will use the buffer later, so we need to scale the market states in the buffer
                self.buffer.update_all(ms_scaler.transform(market_states), 0)
                self.buffer.update_all(ms_scaler.transform(nmarket_states), 4)
            
            # Step 3: Update the network every several steps
            if i >= self.config.agent_learn_start and i % self.config.agent_network_update_freq == 0:
                
                # sample a batch from the replay buffer
                bsm, bsp, ba, br, bnsm, bnsp, bd = self.buffer.sample(self.config.agent_batch_size)

                market_states = self._to_tensor(self._filter(ms_scaler.transform(np.array(bsm))))
                private_states = self._to_tensor(np.array(bsp))
                actions = self._to_tensor(np.array(ba), dtype=torch.long)
                rewards = self._to_tensor(np.array(br))
                nmarket_states = self._to_tensor(self._filter(ms_scaler.transform(np.array(bnsm))))
                nprivate_states = self._to_tensor(np.array(bnsp))
                masks = self._to_tensor(1 - np.array(bd) * 1)
                nactions = self.network(nmarket_states, nprivate_states).argmax(1)

                Qtarget = (rewards + masks * self.config.agent_gamma * \
                    self.network_target(nmarket_states, nprivate_states)[range(self.config.agent_batch_size), \
                    nactions]).detach()
                Qvalue = self.network(market_states, private_states)[range(self.config.agent_batch_size), actions]
                avg_Q = Qvalue.mean().detach()
                loss = self.loss_func(Qvalue, Qtarget)
                self.network.zero_grad()
                loss.backward()
                for param in self.network.parameters():
                    param.grad.data.clamp_(-1, 1)
                # print('Finish the {}-th iteration, the loss = {}'.format(i, float(loss)))
                self.optimizer.step()
                self.scheduler.step()
                
            # Step 4: Update target network
            if i % self.config.agent_target_update_freq == 0:
                self.network_target.load_state_dict(self.network.state_dict())

            # Step 5: Evaluate and log performance
            if i % self.config.agent_plot_freq == 0 and len(train_record) > 0:
                eval_agent = (lambda sm, sp: self.network.act_egreedy(ms_scaler.transform(sm.reshape(1, -1)).flatten(), sp, e=0.0)) \
                    if i > self.config.agent_learn_start else \
                    (lambda sm, sp: self.network.act_egreedy(sm, sp, e=0.0))
                self.evaluation.evaluate_detail_batch(eval_agent, iteration=i)
                print(train_record[-1])

            if i % self.config.agent_eval_freq == 0:
                eval_agent = (lambda sm, sp: self.network.act_egreedy(ms_scaler.transform(sm.reshape(1, -1)).flatten(), sp, e=0.0)) \
                    if i > self.config.agent_learn_start else \
                    (lambda sm, sp: self.network.act_egreedy(sm, sp, e=0.0))
                eval_record.append(self.evaluation.evaluate(eval_agent))
                print(eval_record[-1])

        return train_record, eval_record


class Evaluation(object):
    def __init__(self, config):
        super(Evaluation, self).__init__()
        self.config = config
        self.env = make_env(config)

    def evaluate(self, agent):
        bp_list = []
        rew_list = []
        for code in self.config.code_list_validation:
            for date in self.config.date_list_validation:
                record = self.evaluate_single(agent, code=code, date=date)
                bp_list.append(record['BP'].values[-1])
                rew_list.append(record['reward'].sum())

        return dict(
            BP=np.mean(bp_list),
            reward=np.mean(rew_list)
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
            action = agent(sm, sp)
            nsm, nsp, reward, done, info = self.env.step(action)

            if self.config.simulation_action_type == 'discrete_pq':
                order_price = self.config.simulation_discrete_actions[action][0]
                order_price = np.round((1 + order_price / 10000) \
                    * self.env.data.obtain_level('askPrice', 1) * 100) / 100
            elif self.config.simulation_action_type == 'discrete_p':
                order_price = self.config.simulation_discrete_actions[action]
                order_price = np.round((1 + order_price / 10000) \
                    * self.env.data.obtain_level('askPrice', 1) * 100) / 100
            elif self.config.simulation_action_type == 'discrete_q':
                order_price = self.env.data.obtain_level('bidPrice', 1)

            record.append(dict(
                code=code,
                date=date,
                step=step,
                quantity=self.env.quantity,
                action=action,
                ask_price=self.env.data.obtain_level('askPrice', 1),
                bid_price=self.env.data.obtain_level('bidPrice', 1),
                order_price=order_price,
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
        return dict(mean_bp_first=mean_bp_first, mean_bp_last=mean_bp_last, mean_rew_first=mean_rew_first, mean_rew_last=mean_rew_last)

def run(argus):

    model, lr, lin_reg, parallel_id = argus

    config = DefaultConfig()
    config.agent_learning_rate = lr
    config.simulation_linear_reg_coeff = lin_reg
    config.agent_network_structrue = model
    info = dict(learning_rate=lr, linear_reg=lin_reg, architecture=model.__name__, parallel_id=parallel_id)

    id_str = '{}_lr{:.1E}_linreg_{:.1E}_{}'.format(model.__name__, lr, lin_reg, parallel_id)
    config.result_path = os.path.join(config.result_path, id_str)
    os.makedirs(config.result_path, exist_ok=True)
    extend_path = lambda x: os.path.join(config.result_path, x)

    agent = Agent(config)
    train_record, eval_record = agent.learn()
    train_record, eval_record = pd.DataFrame(train_record), pd.DataFrame(eval_record)
    train_record.to_csv(extend_path('dqn_train_record.csv'))
    eval_record.to_csv(extend_path('dqn_eval_record.csv'))
    train_info = Figure().plot_training_process_basic(train_record, extend_path('dqn_train_record.png'))
    eval_info = Figure().plot_training_process_basic(eval_record, extend_path('dqn_eval_record.png'))
    info.update({('trn_' + k): v for k, v in train_info.items()})
    info.update({('val_' + k): v for k, v in eval_info.items()})

    return info

if __name__ == '__main__':

    record = []
    test_list = list(itertools.product(
        [MLPNetwork_complex, MLPNetwork_Xcomplex], 
        [2e-5, 1e-5, 5e-6], 
        [0.1, 0.01], 
        np.arange(5)
    ))

    pool = Pool(4)
    record = pool.map(run, test_list)
    record = pd.DataFrame(record)
    record.to_csv(os.path.join(DefaultConfig().result_path, 'result_original.csv'))
    stats = record.groupby(['learning_rate', 'linear_reg', 'architecture']).agg([np.mean, np.std])
    stats.to_csv(os.path.join(DefaultConfig().result_path, 'result_stats.csv'))
