import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
from typing import Union

Array = np.ndarray


class FakeMarketSimulator:
    def __init__(self):
        size = 512

        close_prices = get_random_array(initial_value=100, size=size, drift_rate=0.01, volatility=0.1)
        self.close_prices = close_prices.round(2)

        volumes = get_random_array(initial_value=1000, size=size, drift_rate=0.01, volatility=0.1)
        self.volumes = np.abs(volumes)

        # map_to_executed_volume =
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


def get_random_array(initial_value: float, size: int, drift_rate: float, volatility: float):
    dt = 1 / size
    delta = rd.normal(loc=(drift_rate - 0.5 * volatility ** 2) * dt, scale=volatility * np.sqrt(dt), size=size)
    return initial_value * np.exp(delta.cumsum())


def tutorial__get_random_array():
    array = get_random_array(initial_value=100, size=512, drift_rate=0.01, volatility=0.1)
    plt.plot(array)
    plt.show()


"""ploy fix"""


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


def run():
    env = FakeMarketSimulator()


if __name__ == '__main__':
    run()
    # tutorial__get_fix_ploy_nums()
