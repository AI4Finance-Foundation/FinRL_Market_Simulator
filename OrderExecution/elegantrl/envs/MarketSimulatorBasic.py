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

        print(self.map_to_executed_volume_rate(np.array([0.0, 0.5, 1.0, 2., 4., 6.])))

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


algo_json = {
    "name": 'Iceberg',
    "price_depth": 0.2,
    "interval": 1 * 1000,
    "start_time": None,  # start_time_timestamp_ms,
    "order_type": "time_preferred",
    "display_vol": 0.05,
    "price_tick_added": 1,
    "reject_interval": 3 * 1000,
    "imb_sum_ratio": 1,
    "imb_level1_ratio": 0.8,
    "traded_interval": 2 * 1000,
    "last_order_kbest": False,
    "allow_pending_up_limit": False,
    "allow_pending_down_limit": True
}

MarketDepth = 5
UnitVolume = 100
UnitPrice = 0.01


class IcebergPolicy:
    def __init__(self):
        self.interval_ms = 1 * 1000
        self.reject_interval_ms = 3 * 1000
        self.traded_interval_ms = 2 * 1000
        self.start_time_timestamp_ms = None

        self.price_ticks = UnitPrice * 0  # only for price_preferred （价格偏移值）
        self.price_depth = 0.2  # assert 0 <= price_depth < 1.
        self.display_vol = 0.05  # assert display_vol > 0  # 要吃掉的订单的 百分比
        self.imb_sum_ratio = 1.0
        self.imb_level1_ratio = 0.8
        self.price_tick_added = 1

        self.last_order_k_best = False
        self.allow_pending_up_limit = False
        self.allow_pending_down_limit = False

        self.if_time_preferred = True  # or price_preferred

        self.if_buy = True  # buy or sell mode

        '''set by self.update_state'''
        self.ask_px1 = None
        self.ask_vol = None
        self.ask_vol_sum = None  # assert self.ask_vol_sum > 0.0

        self.bid_px1 = None
        self.bid_vol = None
        self.bid_vol_sum = None  # assert self.bid_vol_sum > 0.0

    def update_state(self, ask_prices, ask_volumes, bid_prices, bid_volumes):
        self.ask_px1 = ask_prices[0]
        self.ask_vol = ask_volumes
        self.ask_vol_sum = self.ask_vol[:MarketDepth].sum()

        self.bid_px1 = bid_prices[0]
        self.bid_vol = bid_volumes
        self.bid_vol_sum = self.bid_vol[:MarketDepth].sum()

    def get_action(self, state):
        pass

    def get_display_volume(self):  #
        if self.display_vol < 1:

            if self.if_buy:
                vol = self.display_vol * self.ask_vol_sum / MarketDepth
            else:
                vol = self.display_vol * self.bid_vol_sum / MarketDepth
            # 某时刻五档行情可能为0
            if vol == 0.0:
                vol = UnitVolume
        else:
            vol = int(self.display_vol)  # 这里表示它不是一个百分比，而是一个实际的要成交的 executed volume
        return vol

    def send_sub_order_without_locked(self):
        # 若母订单被取消，或者已经完成母订单任务了，就直接 return

        if_timeout = bool(None)  # 获取系统本地的时间，检测传过来的时间

        if self.ask_vol_sum == 0:  # 表示现在是涨停，订单就挂 bid1
            price = self.bid_px1
        elif self.bid_vol_sum == 0:  # 跌停
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

        # SendSubOrder
        self.volume = None
        vol = self.get_display_volume()
        self.volume -= vol  # 先假装发出去的子订单已经全部成交了，得到一个 self.volume 用于计算下一个

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


if __name__ == '__main__':
    # run()
    # tutorial__get_fix_ploy_nums()
    tutorial__get_random_array()
