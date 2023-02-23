import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
from typing import Union, Tuple

Array = np.ndarray


class MarketSynthetics:
    def __init__(self):
        size = 512

        close_prices = 100 * generate_brownian_motion_ary(size=size, drift_rate=0.01, volatility=0.1)
        self.close_prices = close_prices.round(2)

        volumes = 1000 * np.exp(generate_arima_noise(ar_params=(0.2, -0.3), ma_params=(0.4, 0.1), size=size))
        self.volumes = np.abs(volumes)

        x_data = np.linspace(start=0, stop=2, num=32)
        y_data = np.exp(-x_data ** 2)  # normal distribution Probability density function (Gaussian integral)
        self.fit_plot_nums = np.polyfit(x_data, y_data, deg=5)

        self.t = 0
        self.max_step = size

    def reset(self):
        self.t = 0
        return self.get_state()

    def get_state(self):
        return np.array((self.close_prices[self.t], self.volumes[self.t]))

    def step(self):
        self.t += 1
        state = self.get_state()
        done = self.t + 1 >= self.max_step
        reward = 0
        return state, reward, done, None

    def map_to_executed_volume_rate(self, x_eval: Union[float, Array]) -> Union[float, Array]:
        y_eval = np.stack([n * x_eval ** i for (i, n) in enumerate(self.fit_plot_nums[::-1])]).sum(axis=0)
        y_eval[x_eval > 2] = 0.
        return y_eval


def generate_brownian_motion_ary(
        size: int = 128,
        drift_rate: float = 0.01,
        volatility: float = 0.1
) -> Array:
    dt = 1 / size
    delta = rd.normal(loc=(drift_rate - 0.5 * volatility ** 2) * dt, scale=volatility * np.sqrt(dt), size=size)
    return np.exp(delta.cumsum())


def generate_arima_noise(
        size: int = 128,
        ar_params: Tuple[float, float] = (0.2, -0.3),
        ma_params: Tuple[float, float] = (0.4, 0.1),
) -> Array:  # autoregressive integrated moving average
    errors = rd.normal(size=size)
    y = np.zeros(size)
    for i in range(max(len(ar_params), len(ma_params)), size):
        ar_term = np.dot(ar_params, y[i - len(ar_params):i][::-1])
        ma_term = np.dot(ma_params, errors[i - len(ma_params):i][::-1])
        y[i] = ar_term + ma_term + errors[i]
    return y


'''IcebergAlgo'''
MarketDepth = 5  # 当前订单执行仿真考虑的订单深度
UnitVolumeSize = 100  # 发到交易所最小单位的订单数量
UnitPrice = 0.01  # 交易所上，最小的价格变化量


class IcebergAlgo:
    def __init__(self):
        """
        在订单执行任务中，当一笔母订单任务被创建出来，它就会根据冰山算法的规则被拆成子订单，提交到市场上，并不断返回母订单的执行结果。
        冰山算法：
        - 根据成交价格或者成交速度 优先的设定，并根据快照数据判断买卖压力，算出执行价格
        - 子订单的订单数量与当前盘口订单数量成固定比例
        - 只有上一笔子订单被执行完毕，才会把早就算好的下一笔子订单提交到交易所（假设某个时刻上一笔订单因为全部成交而结束，而非因为超时撤单而结束）
        """
        self.interval_ms = 1 * 1000  # 获取行情的间隔时间
        self.reject_interval_ms = 3 * 1000  # 订单被拒绝后等待的时间间隔
        self.traded_interval_ms = 2 * 1000  # 检测当前母订单完全执行完毕后，重新获取新一个母订单 的 检测时间间隔？
        self.start_time_timestamp_ms = None  #

        self.price_ticks = UnitPrice * 0  # only for price_preferred （价格偏移值）
        self.imb_sum_ratio = 1.0
        self.imb_level1_ratio = 0.8
        self.price_tick_added = 1

        self.last_order_k_best = False
        self.allow_pending_up_limit = False
        self.allow_pending_down_limit = False

        self.if_time_preferred = True  # or price_preferred
        self.if_buy = True  # buy or sell mode

        '''update by self.get_state()'''
        self.ask_px1 = None
        self.ask_vol = None
        self.ask_vol_sum = None  # assert self.ask_vol_sum > 0.0
        self.bid_px1 = None
        self.bid_vol = None
        self.bid_vol_sum = None  # assert self.bid_vol_sum > 0.0

        '''set by self.reset()'''
        self.timer = None  # 计算从母订单被创建，到当前经过了多长时间，总耗时 += 分步耗时
        self.order_size = None  # 母订单需要被执行掉的订单数量。达到这个目标订单数量后，母订单执行完成。
        self.hidden_size = None  # 母订单还没有被执行掉的订单数量。hidden_size 变为0后，母订单执行完成。
        # visible_size = None  # 母订单被拆分成子订单，子订单被提交到的交易所，是可见的。

        self.market_depth = 0.2  # assert 0. <= price_depth < 1.
        self.visible_ratio = 0.05  # assert 0. < display_vol < 1.  # 要吃掉的订单的 百分比

    def reset(self, order_size: int):
        self.timer = 0  # update in self.step
        self.order_size = order_size
        self.hidden_size = order_size

    def step(self, timer_add: float = 1.0):  # todo
        self.timer += timer_add

    def get_state(self, ask_prices, ask_volumes, bid_prices, bid_volumes):
        self.ask_px1 = ask_prices[0]
        self.ask_vol = ask_volumes
        self.ask_vol_sum = self.ask_vol[:MarketDepth].sum()

        self.bid_px1 = bid_prices[0]
        self.bid_vol = bid_volumes
        self.bid_vol_sum = self.bid_vol[:MarketDepth].sum()

    def get_visible_size(self):
        if self.if_buy:
            size = self.visible_ratio * self.ask_vol_sum / MarketDepth
        else:
            size = self.visible_ratio * self.bid_vol_sum / MarketDepth
        return max(UnitVolumeSize, size)

    def get_visible_price(self, if_timeout: bool):
        if self.ask_px1 == 0 and self.bid_px1 > 0:  # 涨停
            price = self.bid_px1
        elif self.bid_px1 == 0 and self.bid_px1 > 0:  # 跌停
            price = self.ask_px1
        elif if_timeout:
            price = self.ask_px1 if self.if_buy else self.bid_px1  # Market-to-limit orders
        elif self.if_time_preferred:
            if self.if_buy:
                if_m2l = (
                        (self.ask_vol_sum / self.bid_vol_sum > self.imb_sum_ratio) and
                        (self.ask_px1 / self.bid_px1 > self.imb_level1_ratio)
                )  # if Market-to-limit orders
                price = self.bid_px1 if if_m2l else self.ask_px1
            else:
                if_m2l = (
                        (self.bid_vol_sum / self.ask_vol_sum > self.imb_sum_ratio) and
                        (self.bid_px1 / self.ask_px1 > self.imb_level1_ratio)
                )  # if Market-to-limit orders
                price = self.ask_px1 if if_m2l else self.bid_px1
        else:  # price_preferred
            price = self.bid_px1 - self.price_ticks if self.if_buy \
                else self.ask_px1 + self.price_ticks
        return price


'''tutorial'''


def tutorial__get_random_array():
    size = 256

    array = 100 * (generate_brownian_motion_ary(size=size, drift_rate=0.01, volatility=0.1) - 1)
    plt.plot(array, label='100*(brownian_motion-1)')
    array = generate_arima_noise(ar_params=(0.2, -0.3), ma_params=(0.4, 0.1), size=size)
    plt.plot(array, label='arima_noise')
    array = np.exp(-1 + generate_arima_noise(size=size, ar_params=(0.2, -0.3), ma_params=(0.4, 0.1)))
    plt.plot(array, label='exp(arima_noise)')

    plt.title('generate random noise')
    plt.legend()
    plt.grid()
    plt.show()


def tutorial__get_fix_ploy_nums():
    x_data = np.linspace(start=0, stop=2, num=32)
    y_data = np.exp(-x_data ** 2)  # normal distribution Probability density function (Gaussian integral)

    degree = 5
    fit_ploy_nums = np.polyfit(x_data, y_data, degree)
    assert len(fit_ploy_nums) == degree + 1

    plt.scatter(x_data, y_data, label='Data')
    x_eval = np.linspace(0, 2, 128)
    y_eval = np.stack([n * x_eval ** i for (i, n) in enumerate(fit_ploy_nums[::-1])]).sum(axis=0)
    plt.plot(x_eval, y_eval, 'r-', label='Quadratic fit')
    plt.legend()
    plt.show()


'''run'''


def run():
    env = MarketSynthetics()
    env.reset()


if __name__ == '__main__':
    run()
    # tutorial__get_fix_ploy_nums()
    # tutorial__get_random_array()
