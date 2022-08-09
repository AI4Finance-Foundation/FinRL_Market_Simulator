"""
Tuned PPO algorithm for optimized trade execution
"""

from env_v2 import make_env
from storage import RolloutStorage
from constants import CODE_LIST, JUNE_DATE_LIST, VALIDATION_DATE_LIST, VALIDATION_CODE_LIST

from sklearn.preprocessing import StandardScaler
from pathos.multiprocessing import ProcessingPool as Pool
from scipy.special import softmax, expit

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch import Tensor
import torch.optim as opt
from tensorboardX import SummaryWriter

from collections import deque
from collections import namedtuple
from os import makedirs as mkdir
from os.path import join as joindir
from tqdm import trange
import numpy as np 
import pandas as pd
import itertools
import argparse
import math 
import time
import os 

time_stamp = str(time.gmtime()[1]) + "-" + \
    str(time.gmtime()[2]) + "-" + str(time.gmtime()[3]) + "-" + \
    str(time.gmtime()[4]) + "-" + str(time.gmtime()[5])

Transition = namedtuple('Transition', ('sm', 'sp', 'value', 'action', 'logproba', 'mask', 'next_sm', 'next_sp', 'reward'))
EPS = 1e-10
# RESULT_DIR = 'results/ppo_exp1' # + time_stamp
# mkdir(RESULT_DIR, exist_ok=True)


# Hyperparameters
parser = argparse.ArgumentParser(description='PlaNet or Dreamer')
parser.add_argument('--arch', type=str, default='v1', choices=['v1', 'v2', 'v2-5', 'v3'])
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--seed', type=int, default=8888)
args_ = parser.parse_args()

class DefaultConfig(object):

    path_raw_data = '/data/execution_data/raw'
    # path_pkl_data = '/data/execution_data/pkl'
    path_pkl_data = '/mnt/execution_data_v2/pkl'
    # path_pkl_data = os.path.expanduser('~/execution_data/pkl')
    result_path = 'results/ppo_exp3'

    code_list = CODE_LIST
    date_list = JUNE_DATE_LIST
    code_list_validation = VALIDATION_CODE_LIST
    date_list_validation = VALIDATION_DATE_LIST

    agent_scale = 1000
    agent_batch_size = 2048
    agent_learn_start = 1000
    agent_gamma = 0.998
    # agent_epsilon = 0.7
    agent_total_steps = 20 * agent_scale
    
    # Smooth L1 loss (SL1) or mean squared error (MSE)
    # agent_loss_type = 'SL1'
    # agent_lr_decay_freq = 2000
    agent_eval_freq = 100
    agent_plot_freq = 50
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
    # Order volume = total volume / simulation_num_shares
    simulation_num_shares = 10
    # Total volume to trade w.r.t. the basis volume
    simulation_volume_ratio = 0.005
    # ############################### END ###############################

    # ############################### Test Parameters ###############################
    # Encourage a uniform liquidation strategy
    simulation_linear_reg_coeff = [0.1, 0.01]
    agent_network_structrue = None
    # ############################### END ###############################

    
    # Stack the features of the previous x bars
    simulation_loockback_horizon = 5
    # Whether return flattened or stacked features of the past x bars
    simulation_do_feature_flatten = True
    simulation_direction = 'sell'
    # If the quantity is not fully filled at the last time step, we place an MO to liquidate and further plus a penalty
    simulation_not_filled_penalty_bp = 2.0
    # Use discrete actions
    simulation_discreate_actions = \
        np.concatenate([[-50, -40, -30, -25, -20, -15], np.linspace(-10, 10, 21), [15, 20, 25, 30, 40, 50]])
    # Scale the price delta if we use continuous actions
    simulation_continuous_action_scale = 10
    # Use 'discrete' or 'continuous' action space?
    simulation_action_type = 'discrete'

    # PPO parameters =====
    # tricks
    agent_learning_rate = [1e-4, 1e-5]
    eps = 1e-5
    clip_param = 0.2
    num_epoch = 4
    num_mini_batch = 32
    value_loss_coef = 0.5
    entropy_coef = 0.01
    max_grad_norm = 0.5
    use_clipped_value_loss = True

    num_steps = 2048
    gae_lambda = 0.95
    use_linear_lr_decay = True

    
    schedule_adam = 'linear'
    schedule_clip = 'linear'
    layer_norm = True
    state_norm = True
    advantage_norm = True
    lossvalue_norm = True
    clip = 0.2
    lamda = 0.97
    # ====================
    seed = 3333


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


class RunningStat(object):
    def __init__(self, shape):
        self._n = 0
        self._M = np.zeros(shape)
        self._S = np.zeros(shape)

    def push(self, x):
        x = np.asarray(x)
        assert x.shape == self._M.shape
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            oldM = self._M.copy()
            self._M[...] = oldM + (x - oldM) / self._n
            self._S[...] = self._S + (x - oldM) * (x - self._M)

    @property
    def n(self):
        return self._n

    @property
    def mean(self):
        return self._M

    @property
    def var(self):
        return self._S / (self._n - 1) if self._n > 1 else np.square(self._M)

    @property
    def std(self):
        return np.sqrt(self.var)

    @property
    def shape(self):
        return self._M.shape


class ZFilter:
    """
    y = (x-mean)/std
    using running estimates of mean,std
    """

    def __init__(self, shape, demean=True, destd=True, clip=10.0):
        self.demean = demean
        self.destd = destd
        self.clip = clip

        self.rs = RunningStat(shape)

    def __call__(self, x, update=True):
        if update: self.rs.push(x)
        if self.demean:
            x = x - self.rs.mean
        if self.destd:
            x = x / (self.rs.std + 1e-8)
        if self.clip:
            x = np.clip(x, -self.clip, self.clip)
        return x

    def output_shape(self, input_space):
        return input_space.shape


class Memory(object):
    def __init__(self):
        self.memory = []

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self):
        return Transition(*zip(*self.memory))

    def __len__(self):
        return len(self.memory)


class FixedCategorical(torch.distributions.Categorical):
  def sample(self):
    return super().sample().unsqueeze(-1)

  def log_probs(self, actions):
    return (
        super()
        .log_prob(actions.squeeze(-1))
        .view(actions.size(0), -1)
        .sum(-1)
        .unsqueeze(-1)
    )

  def mode(self):
    return self.probs.argmax(dim=-1, keepdim=True)


class Categorical(nn.Module):
  def __init__(self, num_inputs, num_outputs):
    super(Categorical, self).__init__()

    def init(module, weight_init, bias_init, gain=1):
      weight_init(module.weight.data, gain=gain)
      bias_init(module.bias.data)
      return module

    init_ = lambda m: init(
      m,
      nn.init.orthogonal_,
      lambda x: nn.init.constant_(x, 0),
      gain=0.01)

    self.linear = init_(nn.Linear(num_inputs, num_outputs))

  def forward(self, x):
    x = self.linear(x)
    return FixedCategorical(logits=x)


class ActorCritic_v2_Discrete(nn.Module):
  def __init__(self, num_inputs1, num_inputs2, num_outputs, hidden=64, layer_norm=True):
    super(ActorCritic_v2_Discrete, self).__init__()

    self.num_inputs1 = num_inputs1
    self.num_inputs2 = num_inputs2
    self.num_outputs = num_outputs

    def init(module, weight_init, bias_init, gain=1):
      weight_init(module.weight.data, gain=gain)
      bias_init(module.bias.data)
      return module

    init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                           constant_(x, 0), np.sqrt(2))
        
    self.actor_fc1 = nn.Sequential(init_(nn.Linear(num_inputs1, hidden*2)), nn.Tanh(),
                                   init_(nn.Linear(hidden*2, hidden)), nn.Tanh())
    self.actor_fc2 = nn.Sequential(init_(nn.Linear(num_inputs2, hidden)), nn.Tanh())
    self.actor_fc3 = nn.Sequential(init_(nn.Linear(hidden*2, hidden)), nn.Tanh())
    
    self.dist = Categorical(hidden, num_outputs)

    self.critic_fc1 = nn.Sequential(init_(nn.Linear(num_inputs1, hidden*2)), nn.Tanh(), 
                                    init_(nn.Linear(hidden*2, hidden)), nn.Tanh())
    self.critic_fc2 = nn.Sequential(init_(nn.Linear(num_inputs2, hidden)), nn.Tanh())
    self.critic_fc3 = nn.Sequential(init_(nn.Linear(hidden*2, hidden)), nn.Tanh())
    self.critic_linear = init_(nn.Linear(hidden, 1))

    self.train()

  def forward(self, market_states, private_states):
    """
    run policy network (actor) as well as value network (critic)
    :param states: a Tensor2 represents states
    :return: 3 Tensor2
    """

    hidden_actor = self._forward_actor(market_states, private_states)
    hidden_critic = self._forward_critic(market_states, private_states)

    critic_value = self.critic_linear(hidden_critic)
    return critic_value, hidden_actor

  def _forward_actor(self, market_states, private_states):
    market = self.actor_fc1(market_states)
    private = self.actor_fc2(private_states)
    states = torch.cat((market, private), 1)  # (1, hidden) + (1, hidden) => (1, hidden * 2)
    hidden_actor = self.actor_fc3(states)

    return hidden_actor

  def _forward_critic(self, market_states, private_states):
    market = self.critic_fc1(market_states)
    private = self.critic_fc2(private_states)
    states = torch.cat((market, private), 1)
    hidden_critic = self.critic_fc3(states)

    return hidden_critic
  
  def act(self, market_states, private_states):
    value, actor_features = self.forward(market_states, private_states)
    dist = self.dist(actor_features)

    action = dist.sample()

    action_log_probs = dist.log_probs(action)

    return value, action, action_log_probs
  
  def get_value(self, market_states, private_states):
    value, _ = self.forward(market_states, private_states)
    return value

  def evaluate_actions(self, market_states, private_states, action):
    value, actor_features = self.forward(market_states, private_states)
    dist = self.dist(actor_features)

    action_log_probs = dist.log_probs(action)
    dist_entropy = dist.entropy().mean()

    return value, action_log_probs, dist_entropy


class Agent(object):

  def __init__(self, config, writer):
    super(Agent, self).__init__()

    self._set_seed()

    # ==== initialization ====
    self.clip_param = config.clip_param
    self.ppo_epoch = config.num_epoch
    self.num_mini_batch = config.num_mini_batch

    self.value_loss_coef = config.value_loss_coef
    self.entropy_coef = config.entropy_coef

    self.max_grad_norm = config.max_grad_norm
    self.use_clipped_value_loss = config.use_clipped_value_loss

    self.num_steps = config.num_steps
    self.use_linear_lr_decay = config.use_linear_lr_decay

    self.config = config
    self.env = make_env(config)
    self.dim_input1 = self.env.observation_dim       # dimension of market states
    self.dim_input2 = 2                              # dimension of private states
    self.dim_output = self.env.action_dim            # for continuous, =1
    
    network = config.agent_network_structrue
    self.network = network(self.dim_input1, self.dim_input2, self.dim_output).to(device=self.config.agent_device)   
    self.optimizer = opt.Adam(self.network.parameters(), lr=config.agent_learning_rate, eps=config.eps)
    # =========================

    # ==== Print Parameters ====
    print("Network:", config.agent_network_structrue)
    print("Learning Rate:", config.agent_learning_rate)
    print("EPS:", config.eps)
    print("Clip param:", self.clip_param)
    print("PPO epoch:", self.ppo_epoch)
    print("Num mini batch:", self.num_mini_batch)
    print("Value loss coef:", self.value_loss_coef)
    print("Entropy coef:", self.entropy_coef)
    print("Max grad norm:",  self.max_grad_norm)
    print("Use clipped value loss:", self.use_clipped_value_loss)
    print("Num steps:", self.num_steps)
    print("use_linear_lr_decay:", self.use_linear_lr_decay)
    # ===========================


    self.rollouts = RolloutStorage(self.num_steps, self.dim_input1, self.dim_input2, self.dim_output)

    self.running_state_m = ZFilter((self.dim_input1,), clip=5.0)
    self.running_state_p = ZFilter((self.dim_input2,), clip=5.0)

    self.writer = writer

    self.evaluation = Evaluation(self.config)
  
  @staticmethod
  def _filter(state):
    return np.clip(state, -3, 3)
  
  def _set_seed(self, seed=None):
    if seed is None:
      seed = int.from_bytes(os.urandom(4), byteorder='little')
    else:
      seed = seed + 1234
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

  def learn(self):
    
    train_record = []
    eval_record = []

    # record average 1-round cumulative reward in every episode
    # reward_record = []
    global_steps = 0

    ms_scaler = StandardScaler()  

    self.env.reset()  # warm up the environment

    # ==== market state normalization ====
    obs_market_list = []
    for _ in range(self.num_steps):
      # random sample action to collect some samples
      a = self.env.action_sample_func()
      obs_market, obs_private, reward, done, info = self.env.step(a)
      if done:
        obs_market, obs_private = self.env.reset()
      obs_market_list.append(obs_market)
      
    ms_scaler.fit(np.array(obs_market_list)) 
    # =====================================

    
    obs_market, obs_private = self.env.reset()

    obs_market = self._filter(ms_scaler.transform(np.array(obs_market).reshape(1, -1)))[0]
    self.rollouts.obs_market[0].copy_(torch.from_numpy(obs_market))
    self.rollouts.obs_private[0].copy_(torch.from_numpy(obs_private))
    self.rollouts.to(self.config.agent_device)

    for i_episode in trange(self.config.agent_total_steps):
      
      reward_list = []
      
      if self.use_linear_lr_decay:
        # decrease learning rate linearly
        lr = self.config.agent_learning_rate - (self.config.agent_learning_rate * (i_episode / float(self.config.agent_total_steps)))
        for param_group in self.optimizer.param_groups:
          param_group['lr'] = lr

      reward_sum = 0
      t = 0

      for step in range(self.num_steps):
        # (1) Sample actions
        with torch.no_grad():
          value, action, action_log_prob = self.network.act(
              self.rollouts.obs_market[step].unsqueeze(0), self.rollouts.obs_private[step].unsqueeze(0))

        # Obser reward and next obs
        obs_market, obs_private, reward, done, info = self.env.step(action)
        obs_market = self._filter(ms_scaler.transform(np.array(obs_market).reshape(1, -1)))[0]

        # If done then clean the history of observations.
        masks = torch.FloatTensor((0.0,)) if done else torch.FloatTensor((1.0,))
        reward = torch.FloatTensor((reward,)) 
        


        reward_sum += reward
        if done:
          train_record.append(dict(
            i=i_episode, 
            reward=reward_sum,
            BP=self.env.get_metric('BP'),
            IS=self.env.get_metric('IS'),
            code=info['code'],
            date=info['date'],
            start_index=info['start_index']
            ))
          reward_list.append(reward_sum)
          global_steps += (t + 1)
          reward_sum = 0
          t = 0

          obs_market, obs_private = self.env.reset()
          obs_market = self._filter(ms_scaler.transform(np.array(obs_market).reshape(1, -1)))[0]
        
        t = t + 1
        self.rollouts.insert(torch.from_numpy(obs_market), torch.from_numpy(obs_private),
                             action[0], action_log_prob[0], value[0], reward, masks)

      # reward_record.append({
      #     'episode': i_episode, 
      #     'steps': global_steps, 
      #     'meanepreward': torch.mean(reward_list)})

      with torch.no_grad():
        next_value = self.network.get_value(
            self.rollouts.obs_market[-1].unsqueeze(0), self.rollouts.obs_private[-1].unsqueeze(0)).detach()


      self.rollouts.compute_returns(next_value[0], self.config.agent_gamma, self.config.gae_lambda)
      
      advantages = self.rollouts.returns[:-1] - self.rollouts.value_preds[:-1]
      advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

      value_loss_epoch = 0
      action_loss_epoch = 0
      dist_entropy_epoch = 0

      for e in range(self.ppo_epoch):
        data_generator = self.rollouts.feed_forward_generator(advantages, self.num_mini_batch)

        for sample in data_generator:
          obs_market_batch, obs_private_batch, actions_batch, \
              value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                  adv_targ = sample

          # Reshape to do in a single forward pass for all steps
          values, action_log_probs, dist_entropy = self.network.evaluate_actions(
              obs_market_batch, obs_private_batch, actions_batch)

          ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
          surr1 = ratio * adv_targ
          surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                              1.0 + self.clip_param) * adv_targ
          action_loss = -torch.min(surr1, surr2).mean()

          if self.use_clipped_value_loss:
              value_pred_clipped = value_preds_batch + \
                  (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
              value_losses = (values - return_batch).pow(2)
              value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
              value_loss = 0.5 * torch.max(value_losses,
                                            value_losses_clipped).mean()
          else:
              value_loss = 0.5 * (return_batch - values).pow(2).mean()

          self.optimizer.zero_grad()
          (value_loss * self.value_loss_coef + action_loss -
            dist_entropy * self.entropy_coef).backward()
          nn.utils.clip_grad_norm_(self.network.parameters(),
                                    self.max_grad_norm)
          self.optimizer.step()

          value_loss_epoch += value_loss.item()
          action_loss_epoch += action_loss.item()
          dist_entropy_epoch += dist_entropy.item()

      num_updates = self.ppo_epoch * self.num_mini_batch

      value_loss_epoch /= num_updates
      action_loss_epoch /= num_updates
      dist_entropy_epoch /= num_updates

      # value_loss_epoch, action_loss_epoch, dist_entropy_epoch
      self.rollouts.after_update()


      # Step 5: Evaluate and log performance
      if i_episode % self.config.agent_plot_freq == 0 and len(train_record) > 0:
        print(train_record[-1])
        self.evaluation.evaluate_detail_batch(self.network, ms_scaler, iteration=i_episode)

        self.writer.add_scalar("train/reward", torch.mean(train_record[-1]['reward']), i_episode)
        self.writer.add_scalar("train/BP", train_record[-1]['BP'], i_episode)
        self.writer.add_scalar("train/IS", train_record[-1]['IS'], i_episode)

        self.writer.add_scalar("train/value_loss_epoch", value_loss_epoch, i_episode)
        self.writer.add_scalar("train/action_loss_epoch", action_loss_epoch, i_episode)
        self.writer.add_scalar("train/dist_entropy_epoch", dist_entropy_epoch, i_episode)

      if i_episode % self.config.agent_eval_freq == 0:
        eval_record.append(self.evaluation.evaluate(self.network, ms_scaler))
        print("BP:", eval_record[-1]['BP'], 'Reward:', eval_record[-1]['reward'])

        np.save(self.config.result_path + "/eval_record_"+str(i_episode)+".npy", eval_record[-1]['ac_list'])

        self.writer.add_scalar("eval/reward", np.mean(eval_record[-1]['reward']), i_episode)
        self.writer.add_scalar("eval/BP", np.mean(eval_record[-1]['BP']), i_episode)

        self.writer.add_scalar("eval/ac_min", np.mean(eval_record[-1]['ac_min']), i_episode)
        self.writer.add_scalar("eval/ac_max", np.mean(eval_record[-1]['ac_max']), i_episode)
        self.writer.add_scalar("eval/ac_mean", np.mean(eval_record[-1]['ac_mean']), i_episode)

    return train_record, eval_record


class Evaluation(object):
  def __init__(self, config):
    super(Evaluation, self).__init__()
    self.config = config
    self.env = make_env(config)

  def evaluate(self, network, scalar):
    bp_list = []
    rew_list = []
    ac_list = []
    ac_mean_list = []
    ac_logstd_list = []
    for code in self.config.code_list_validation:
        for date in self.config.date_list_validation:
            record, action_list, action_mean_list, action_logstd_list = self.evaluate_single(network, scalar, code=code, date=date)
            bp_list.append(record['BP'].values[-1])
            rew_list.append(record['reward'].sum())
            ac_list.append(action_list)
            ac_mean_list.append(action_mean_list)
            ac_logstd_list.append(action_logstd_list)

    return dict(
        BP=np.mean(bp_list),
        reward=np.mean(rew_list),
        ac_min = np.min(ac_list),
        ac_max = np.max(ac_list),
        ac_mean = np.mean(ac_list),
        ac_list = ac_list
    )

  def evaluate_detail_batch(self, network, scalar, iteration=1,
    code='000504.XSHE', 
    date_list=['2021-06-01', '2021-06-03', '2021-06-04', '2021-07-02', '2021-07-05', '2021-07-06']):

    path = os.path.join(self.config.result_path, 'evaluation', 'it{:08d}'.format(iteration))
    os.makedirs(path, exist_ok=True)

    record = []
    for date in date_list:
        for i in range(5):
            res, _, _, _ = self.evaluate_single(network, scalar, code=code, date=date)
            record.append(res)
            Figure().plot_policy(df=res, filename=os.path.join(path, 'fig_{}_{}_{}.png'.format(code, date, i)))

    pd.concat(record).to_csv(os.path.join(path, 'detail_{}.csv'.format(code)))

  def evaluate_single(self, network, scalar, code='600519.XSHG', date='2021-06-01'):
    record = []
    sm, sp = self.env.reset(code, date)
    done = False 
    step = 0
    action = None 
    info = dict(status=None)

    action_list = []
    action_mean_list = []
    action_logstd_list = []

    while not done:
      sm = Agent._filter(scalar.transform(sm.reshape(1, -1)))[0]
      value, action, action_log_prob = network.act(Tensor(sm).unsqueeze(0).to(device=self.config.agent_device),
                                                   Tensor(sp).unsqueeze(0).to(device=self.config.agent_device))
      action = action.item()
      action_list.append(action)
      action_logstd_list.append(action_log_prob.item())

      nsm, nsp, reward, done, info = self.env.step(action)

      record.append(dict(
          code=code,
          date=date,
          step=step,
          quantity=self.env.quantity,
          action=action,
          ask_price=self.env.data.obtain_level('askPrice', 1),
          bid_price=self.env.data.obtain_level('bidPrice', 1),
          order_price=np.round((1 + self.config.simulation_discreate_actions[action] / 10000) \
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

    return pd.DataFrame(record), action_list, action_mean_list, action_logstd_list


def run(argus):

    model, lr, lin_reg, num_epoch, parallel_id = argus

    config = DefaultConfig()
    config.agent_learning_rate = lr
    config.simulation_linear_reg_coeff = lin_reg
    config.num_epoch = num_epoch
    # config.simulation_continuous_action_scale = action_scale
    
    # config.agent_network_structrue = model
    if model == 'v2-5':
      print("discrete ppo")
      config.agent_network_structrue = ActorCritic_v2_Discrete
    # elif model == 'v3':
    #   config.agent_network_structrue = ActorCritic_v3
    else:
      raise NotImplementedError
    info = dict(learning_rate=lr, linear_reg=lin_reg, num_epoch=num_epoch, architecture=config.agent_network_structrue.__name__, parallel_id=parallel_id)

    print("Config:", info)

    id_str = '{}_lr-{:.1E}_linreg-{:.1E}_numepoch-{}_id-{}'.format(model, lr, lin_reg, num_epoch, parallel_id)
    config.result_path = os.path.join(config.result_path, id_str)
    print("result path:", config.result_path)
    os.makedirs(config.result_path, exist_ok=True)
    extend_path = lambda x: os.path.join(config.result_path, x)

    writer = SummaryWriter(config.result_path + '/logs-' + str(parallel_id))
    agent = Agent(config, writer)
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
    # test_list = list(itertools.product(['v1', 'v2', 'v3'], [3e-4, 1e-4], [0.1, 0.01], [3, 5, 10], np.arange(5)))
    test_list = list(itertools.product(['v2-5',], [5e-5], [0.01,], [4,], np.arange(3)))

    pool = Pool(3)
    record = pool.map(run, test_list)
    record = pd.DataFrame(record)
    record.to_csv(os.path.join(DefaultConfig().result_path, 'result_original.csv'))
    stats = record.groupby(['learning_rate', 'linear_reg', 'architecture']).agg([np.mean, np.std])
    stats.to_csv(os.path.join(DefaultConfig().result_path, 'result_stats.csv'))
    