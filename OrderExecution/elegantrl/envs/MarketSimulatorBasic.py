import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
from typing import Union, Tuple

Array = np.ndarray


class FakeMarketSimulator:
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


# class IcebergPolicy:
#     def __init__(self):
        

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
    env = FakeMarketSimulator()


if __name__ == '__main__':
    # run()
    # tutorial__get_fix_ploy_nums()
    tutorial__get_random_array()
