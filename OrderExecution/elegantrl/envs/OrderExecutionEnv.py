import os
import torch
from random import shuffle
from functorch import vmap
from shares_data_process import get_share_dicts_by_day


class OrderExecutionVecEnv:
    def __init__(self, num_envs: int = 4, gpu_id: int = 0, if_random=False,
                 share_name: str = '000768_XSHE', beg_date: str = '2022-09-01', end_date: str = '2022-09-03', ):
        self.if_random = if_random  # 设计随机的 reset，能让策略在更多样的state下学习，会提高策略泛化能力。
        self.num_levels = 5  # 从10档行情中，选出 n_levels 个档位 用于仿真
        self.price_scale = 25  # 策略网络输出的第一个动作特征，是订单的卖出价格与上一时刻的变化量，表示30个档位
        self.volume_scale = 1e-2  # 自动设置订单执行任务里，需要被执行的订单的数量，是成交量的 volume_scale 倍
        self.executed_scale = 2e-2  # last_price 里，订单的成交比率
        assert self.volume_scale < self.executed_scale

        '''stack state'''
        self.n_stack = 8  # 保存 n_stack 个不同时刻t的state，用于堆叠state
        self.n_state = []  # 保存 n_stack 个不同时刻t的state，用于堆叠state

        '''device'''
        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")
        # 为Vectorized Env 指定单张GPU设备进行并行计算

        '''load data'''
        self.max_len = None  # 赋值为None，会在不调用env.reset() 就运行step的情况下，会主动引发可预见的错误
        self.share_name = share_name  # 当前被随机抽取出来的股票的名字
        self.cumulative_returns = torch.zeros(0)  # 赋值为 torch.zeros(0) 这个空张量，是有意强调他们是张量，而不是None

        self.price = torch.zeros(0)
        self.volume = torch.zeros(0)
        self.ask_prices = torch.zeros(0)  # indices=[1, max_level] 各个级别的成交价
        self.bid_prices = torch.zeros(0)  # indices=[1, max_level] 各个级别的成交价
        self.ask_volumes = torch.zeros(0)  # indices=[1, max_level] 各个级别的成交量
        self.bid_volumes = torch.zeros(0)  # indices=[1, max_level] 各个级别的成交量
        self.tech_factors = torch.zeros(0)
        self.total_quantity = torch.zeros(0)  # 订单执行的目标成交量（希望在一天内达成这个目标成交量）

        self.data_dicts = self.load_share_data_dicts(
            data_dir='./shares_data_by_day', share_name=share_name,
            beg_date=beg_date, end_date=end_date)

        '''reset'''
        self.t = 0  # 时刻t
        self.cash = torch.zeros(0)  # 现金，不需要加入的state里，因为我们只卖出，不买入
        self.quantity = torch.zeros(0)  # 基础成交量
        self.total_asset = torch.zeros(0)  # 总资产，总资产等于现金+商品折算为现金。（在订单执行任务里，商品折算为0现金）
        self.remain_quantity = torch.zeros(0)  # 剩余成交量，智能体需要就是把商品都卖出，让它在最后变成0

        '''env info'''
        self.env_name = 'OrderExecutionVecEnv-v2'
        self.num_envs = num_envs
        self.max_step = max([data_dict['max_len'] for data_dict in self.data_dicts])  # 选取数据中最长的步数作为 max_step
        self.state_dim = (4 + self.data_dicts[0]['tech_factors'].shape[1]) * self.n_stack
        self.action_dim = 2
        self.if_discrete = False

        '''function for vmap'''
        self.inplace_cash_quantity = vmap(
            func=self._inplace_cash_quantity, in_dims=(0, 0, 0, None, None), out_dims=0
        )

        self._get_state = vmap(
            func=lambda remain_quantity, quantity, remain_step_rate, last_price, tech_factor:
            torch.hstack((remain_quantity, quantity, remain_step_rate, last_price, tech_factor)),
            in_dims=(0, 0, None, None, None), out_dims=0
        )

        '''def get_data_dict'''
        self.rand_id = 0
        shuffle(self.data_dicts)

    def get_data_dict(self):
        self.rand_id += 1
        if self.rand_id >= len(self.data_dicts):
            self.rand_id = 0
            shuffle(self.data_dicts)
        return self.data_dicts[self.rand_id]  # data_dict

    def reset(self):
        self.t = 0

        '''load data from data_dict to device'''
        data_dict = self.get_data_dict()
        self.max_len = data_dict['max_len']
        self.volume = data_dict['volume'].to(self.device)
        self.price = data_dict['last_price'].to(self.device)
        self.ask_prices = data_dict['ask_prices'].to(self.device)
        self.bid_prices = data_dict['bid_prices'].to(self.device)
        self.ask_volumes = data_dict['ask_volumes'].to(self.device)
        self.bid_volumes = data_dict['bid_volumes'].to(self.device)
        self.tech_factors = data_dict['tech_factors'].to(self.device)

        total_quantity = data_dict['total_quantity'].to(self.device)
        total_quantity = total_quantity.repeat(self.num_envs)

        '''build internal state: cash'''
        self.cash = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.total_asset = self.cash.clone()  # 总资产，总资产等于现金+商品折算为现金。（在订单执行任务里，商品折算为0现金）

        '''build internal state: quantity'''
        self.quantity = total_quantity * self.executed_scale / self.max_len
        total_quantity_scale = torch.arange(self.num_envs).to(self.device) / self.num_envs
        total_quantity_scale = total_quantity_scale * 0.9 + 0.1  # range in [0.1, 0.9]
        self.total_quantity = total_quantity * self.volume_scale * total_quantity_scale
        if self.if_random:
            self.quantity *= torch.rand_like(self.quantity) * 0.2 + 0.9  # range in [0.9, 1.1]
            self.total_quantity *= torch.rand_like(self.total_quantity) * 0.2 + 0.9  # range in [0.9, 1.1]

        self.total_quantity = torch.round(self.total_quantity)
        self.remain_quantity = torch.zeros_like(self.cash) + self.total_quantity

        '''stack state'''
        state = self.get_state()
        self.n_state = [state, ] * 24
        return self.get_n_state()

    def step(self, action):
        self.t += 1
        done = self.t == self.max_len

        '''action'''  # 对策略输出的-1.0~+1.0 的动作进行线性变化，得到仿真环境实际需要的 挂单价格 + 挂单数量
        curr_price = self.get_curr_price(action[:, 0])
        curr_quantity = self.get_curr_quantity(action[:, 1])
        prev_quantity = curr_quantity.clone()

        '''executed in current step immediately'''
        for level in range(self.num_levels):
            self.inplace_cash_quantity(self.cash, curr_quantity, curr_price,
                                       self.bid_prices[level, self.t], self.bid_volumes[level, self.t])

        '''executed in next step'''
        if not done:
            self.inplace_cash_quantity(self.cash, curr_quantity, curr_price,
                                       self.price[self.t + 1], self.volume[self.t + 1] * self.executed_scale)

        '''update remain_quantity'''
        diff_quantity = curr_quantity - prev_quantity
        self.remain_quantity += diff_quantity

        '''get (state, reward, done)'''
        total_asset = self.cash
        reward = (total_asset - self.total_asset) * 2 ** -14
        self.total_asset = self.cash.clone()

        # state = self.reset() if done else self.get_state()  # after self.t += 1
        if done:
            self.cumulative_returns = total_asset / (self.total_quantity * self.price.mean()) * 100  # 100%
            n_state = self.reset()
        else:
            state = self.get_state()
            self.n_state.append(state)
            del self.n_state[0]
            n_state = self.get_n_state()

        done = torch.tensor(done, dtype=torch.bool, device=self.device).expand(self.num_envs)
        return n_state, reward, done, {}

    def get_state(self):  # 得到智能体观测的状态
        return self._get_state(self.remain_quantity / self.total_quantity,
                               self.quantity / self.total_quantity,
                               self.get_tensor(1 - self.t / self.max_len),  # remain_step_rate
                               self.price[self.t] * 2 ** -3,
                               self.tech_factors[self.t])

    def get_n_state(self):
        return torch.hstack([self.n_state[i] for i in (-1, -2, -3, -5, -7, -11, -15, -24)])

    def get_tensor(self, ary):
        return torch.tensor(ary, dtype=torch.float32, device=self.device)

    def get_curr_price(self, action_price):
        delta_price = action_price * (self.price_scale * 0.01)
        return self.price[self.t - 1] + delta_price  # after self.t += 1

    def get_curr_quantity(self, action_quantity):
        quantity_ratio = action_quantity + 1
        curr_quantity = torch.round(quantity_ratio * self.quantity)
        curr_quantity = torch.min(torch.stack((self.remain_quantity, curr_quantity)), dim=0)[0]
        return curr_quantity

    @staticmethod
    def _inplace_cash_quantity(cash, quantity, price, ask_price, ask_volume):
        executed_volume = torch.min(quantity, ask_volume) * (price >= ask_price)
        # 乘以 (price >= ask_price)，相当于一个if，如果是False，那么 execute_volume 相当于是 0，等价于不执行这里的代码
        # 进行这种处理，是因为 vmap 现阶段（2022-11-09）无法加速含有逻辑分支的代码，只能加速静态的代码
        cash += executed_volume * price
        quantity -= executed_volume
        return torch.empty(0)

    @staticmethod
    def get_tech_factors(volume, price, value,
                         ask_prices, ask_volumes,
                         bid_prices, bid_volumes):
        """
        我随便写的根据 ask-bid 数据得到 特征的代码，用GPU计算，有微弱的效果
        用于能检测仿真环境加入 technical factors 的模块是否正常运行
        以后需要替换成更加专业的 technical factors
        """
        ask_values = ask_prices * ask_volumes
        bid_values = bid_prices * bid_volumes

        mean_price = value / volume
        delta_price = price - mean_price

        ask_cum_values = torch.cumsum(ask_values, dim=0)
        bid_cum_values = torch.cumsum(bid_values, dim=0)

        ask_cum_volumes = torch.cumsum(ask_volumes, dim=0)
        bid_cum_volumes = torch.cumsum(bid_volumes, dim=0)

        ask_cum_prices = ask_cum_values / ask_cum_volumes
        del ask_cum_values, ask_cum_volumes
        bid_cum_prices = bid_cum_values / bid_cum_volumes
        del bid_cum_values, bid_cum_volumes

        v_adj_spreads = ask_cum_prices - bid_cum_prices
        del ask_cum_prices, bid_cum_prices

        '''normalization'''
        tech_factors = torch.cat((
            get_ts_trends(value * 2 ** -14, win_size=6, gap_size=6),
            get_ts_trends(value * 2 ** -14, win_size=12, gap_size=8),
            get_ts_trends(mean_price * 2 ** 3, win_size=6, gap_size=6),
            get_ts_trends(mean_price * 2 ** 3, win_size=12, gap_size=8),
            get_ts_trends(delta_price * 2 ** 9, win_size=6, gap_size=6),
            get_ts_trends(delta_price * 2 ** 9, win_size=12, gap_size=8),
            get_ts_trends(v_adj_spreads[0] * 2 ** 6, win_size=6, gap_size=6),
            get_ts_trends(v_adj_spreads[1] * 2 ** 6, win_size=8, gap_size=6),
            get_ts_trends(v_adj_spreads[2] * 2 ** 6, win_size=8, gap_size=8),
            get_ts_trends(v_adj_spreads[3] * 2 ** 6, win_size=12, gap_size=8),
            get_ts_trends(v_adj_spreads[4] * 2 ** 6, win_size=12, gap_size=12),
        ), dim=1)
        torch.nan_to_num_(tech_factors, nan=0.0, posinf=0.0, neginf=0.0)
        return tech_factors

    def load_share_data_dicts(self, data_dir="./data",
                              share_name: str = '000768_XSHE',
                              beg_date='2022-09-01',
                              end_date='2022-09-30'):
        assert share_name in {'000768_XSHE', '000685_XSHE'}
        share_dir = f"{data_dir}/{share_name}"
        share_dicts = get_share_dicts_by_day(share_dir=share_dir, share_name=share_name,
                                             beg_date=beg_date, end_date=end_date,
                                             n_levels=self.num_levels, n_days=5, device=self.device)
        for share_dict in share_dicts:
            for key, value in share_dict.items():
                if isinstance(value, torch.Tensor):
                    share_dict[key] = value.to(torch.device('cpu'))

        data_dicts = []  # 把不同股票的数据放在字典里，reset的时候会随机选择一只股票的数据，加载到GPU里，开始训练
        print('| OrderExecutionEnv data pre processing:', share_name)

        for i, share_dict in enumerate(share_dicts):
            share_name = share_dict['share_name']
            trade_date = share_dict['trade_date']
            print(end=f'{trade_date}  ')
            print() if i % 8 == 7 else None

            # 对这些订单流数据进行处理后，我们能得到一段时间内的 ask 和 bid 快照数据
            ask_volumes = share_dict['ask_volumes']  # 各个级别的成交量
            bid_volumes = share_dict['bid_volumes']  # 各个级别的成交量
            ask_prices = share_dict['ask_prices']  # 各个级别的成交量
            bid_prices = share_dict['bid_prices']  # 各个级别的成交量
            volume = share_dict['volume']  # delta volume 成交的订单数量
            price = share_dict['price']  # last price 最后成交价格
            value = share_dict['value']  # delta value 成交金额总量，换手额度

            tech_factors = self.get_tech_factors(volume, price, value,
                                                 ask_prices, ask_volumes,
                                                 bid_prices, bid_volumes)

            # 先保存到内存里，reset的时候才加载到GPU
            data_dict = {
                'share_name': share_name,
                'max_len': price.shape[0] - 1,
                'total_quantity': volume.sum(),

                'volume': volume,
                'last_price': price,
                'ask_prices': ask_prices,
                'bid_prices': bid_prices,
                'ask_volumes': ask_volumes,
                'bid_volumes': bid_volumes,
                'tech_factors': tech_factors,
            }
            data_dicts.append(data_dict)
        return data_dicts


class OrderExecutionVecEnvForEval(OrderExecutionVecEnv):
    def __init__(self, num_envs: int = 4, gpu_id: int = 0, if_random=False,
                 beg_date: str = '2022-09-01', end_date: str = '2022-09-03', share_name='000685_XSHE'):
        OrderExecutionVecEnv.__init__(self, num_envs=num_envs, gpu_id=gpu_id, if_random=if_random,
                                      beg_date=beg_date, end_date=end_date, share_name=share_name)

        self.curr_price = None
        self.curr_quantity = None
        self.cumulative_returns_days = []

    def reset(self):
        self.rand_id = 0
        self.cumulative_returns_days = []
        return super().reset()

    def step(self, action):  # modified_mark
        n_state, reward, done, info_dict = super().step(action)

        if done[0]:  # modified_mark
            self.cumulative_returns_days.append(self.cumulative_returns)
            self.cumulative_returns = torch.stack(self.cumulative_returns_days).mean(dim=0)

            data_dict = self.data_dicts[self.rand_id]
            self.bid_prices = data_dict['bid_prices'].to(self.device)  # ForPlot
            self.bid_volumes = data_dict['bid_volumes'].to(self.device)  # ForPlot
        return n_state, reward, done, info_dict

    def get_curr_price(self, action_price):
        self.curr_price = super().get_curr_price(action_price)
        return self.curr_price

    def get_curr_quantity(self, action_quantity):
        self.curr_quantity = super().get_curr_quantity(action_quantity)
        return self.curr_quantity


'''get_tech_factors'''


def get_re_cum_sum(ten):
    cum_sum = torch.cumsum(ten, dim=0)
    return ten - cum_sum + cum_sum[-1:None]


def get_all_cum_sum(level_tensors):
    level_cum = level_tensors.clone()
    for i in range(1, level_tensors.shape[1]):
        level_cum[i] += level_cum[i - 1]
    return level_cum


def get_ts_avg_std(ten, win_size=6):  # could be higher performance
    avg = torch.zeros_like(ten)
    std = torch.zeros_like(ten)
    for i in range(win_size, avg.shape[0]):
        tmp = ten[i - win_size:i]
        avg[i] = tmp.mean(dim=0)
        std[i] = tmp.std(dim=0)
    return avg, std


def get_ts_diff(ten, gap_size=6):
    out = torch.zeros_like(ten)
    out[gap_size:] = ten[gap_size:] - ten[:-gap_size]
    return out


def get_ts_trends(ten, win_size=6, gap_size=6):
    avg, std = get_ts_avg_std(ten, win_size)
    avg_diff = get_ts_diff(avg, gap_size)
    std_diff = get_ts_diff(std, gap_size)
    return torch.stack((avg, avg_diff, std, std_diff), dim=1)


"""run"""


def check_with_twap():
    num_envs = 2
    share_name = ['000768_XSHE', '000685_XSHE'][0]
    beg_date = '2022-09-01'
    end_date = '2022-09-01'

    # env = OrderExecutionVecEnv(num_envs=num_envs, gpu_id=0, if_random=False,
    #                            share_name=share_name, beg_date=beg_date, end_date=end_date)
    env = OrderExecutionMinuteVecEnv(num_envs=num_envs, gpu_id=0, if_random=False,
                                     share_name=share_name, beg_date=beg_date, end_date=end_date)
    env.reset()

    action = torch.zeros((num_envs, env.action_dim), dtype=torch.float32, device=env.device)
    # 0: the delta price is 0 in default
    # 1: the quantity scale is +1 in default

    cumulative_rewards = torch.zeros(num_envs, dtype=torch.float32, device=env.device)
    for i in range(env.max_step):
        state, reward, done, _ = env.step(action)
        cumulative_rewards += reward

        if i % 64 == 0:
            env_cumulative_rewards = env.total_asset / env.total_quantity
            print(f"{i:8}  {str(env_cumulative_rewards):64}  {env.remain_quantity}  {reward}")

    print(env.total_asset / env.total_quantity)
    print(env.total_asset)
    print(env.remain_quantity)

    print(f'cumulative_returns {env.cumulative_returns.mean():9.3f}  {env.cumulative_returns.std(dim=0):9.3f}')
    print(f'cumulative_rewards {cumulative_rewards.mean():9.3f}  {cumulative_rewards.std(dim=0):9.3f}')


if __name__ == '__main__':
    check_with_twap()
