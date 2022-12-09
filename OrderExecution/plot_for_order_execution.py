import os
import torch
from OrderExecutionEnv import OrderExecutionVecEnvForEval

"""run"""


def check__ask_price_volume():
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # OMP: Error #15: Initializing libiomp5md.dll

    import matplotlib.pyplot as plt
    import numpy as np

    num_envs = 2
    env = OrderExecutionVecEnvForEval(num_envs=num_envs, beg_date='2022-09-09', end_date='2022-09-09')
    env.if_random = False
    env.reset()

    max_len1 = env.max_len + 1  # after env.reset()
    xs = np.arange(max_len1)
    print('xs.shape', xs.shape)

    '''ask bid price (from level 1 to 5)'''
    from matplotlib.cm import get_cmap
    color_map = get_cmap('bwr')  # Blue White Red, input 0.0 ~ 1.0 or 0 ~ 1000

    ask_prices = np.array(env.ask_prices)
    ask_prices[ask_prices < 7.0] = 7.4  # todo 每天快结束的时候，总有一些成交量特别低的异常数据，因此把它们都赋值为最后一个正常的数值
    print('ask_prices.shape', ask_prices.shape)
    n_level, max_len1 = ask_prices.shape
    for i in range(n_level):  # todo 这里的代码，把 askPrices 画出来
        face_color = color_map(float(1 - i / n_level) * 0.2 + 0.2)  # todo 使用蓝色渐变
        if i + 1 == n_level:
            plt.fill_between(xs, ask_prices[i], np.zeros_like(ask_prices[i]) + np.nanmax(ask_prices[i]),
                             facecolor=face_color)
        else:
            plt.fill_between(xs, ask_prices[i], ask_prices[i + 1],
                             facecolor=face_color)

    bid_prices = np.array(env.bid_prices)
    bid_prices[bid_prices < 1] = np.nan
    print('bid_prices.shape', bid_prices.shape)
    n_level, max_len1 = bid_prices.shape
    for i in range(n_level):  # todo 这里的代码，把 askPrices 画出来
        # face_color = color_map(float(i / n_level) * 0.3 + 0.5 + 0.1) # todo 使用红色渐变
        face_color = color_map(float(1 - i / n_level) * 0.2 + 0.2)  # todo 使用蓝色渐变
        if i + 1 == n_level:
            plt.fill_between(xs, bid_prices[i], np.zeros_like(bid_prices[i]) + np.nanmin(bid_prices[i]),
                             facecolor=face_color)
        else:
            plt.fill_between(xs, bid_prices[i], bid_prices[i + 1],
                             facecolor=face_color)

    last_price = np.array(env.last_price)
    plt.plot(xs, last_price, color='blue', label='last price')  # todo 用蓝色把 last price 画出来

    '''policy: VWAP (using the data in future)'''
    actions = torch.zeros((max_len1, num_envs, 2), dtype=torch.float32, device=env.device)
    print('actions.shape', actions.shape)
    volume_weights = (env.volume / env.volume.mean() - 1) / env.volume.std(dim=0) + 1

    k = 5  # 平滑操作，卷积核是 k*2+1=11
    volume_smooths = volume_weights.clone()
    for i in range(1, k):
        volume_smooths[i:] += volume_weights[:-i]
        volume_smooths[:-i] += volume_weights[i:]
    volume_smooths /= 2 * k - 1  # convolve
    volume_smooths[:k] = volume_smooths[k]
    volume_smooths[-k:] = volume_smooths[-k]

    prev_price = env.last_price.clone()
    prev_price[1:] = env.last_price[:-1]
    curr_price = env.last_price * ((volume_smooths - 1.0) * 16 + 1.0)
    curr_price = torch.round(curr_price * 100) / 100
    curr_price = torch.min(torch.stack((curr_price, env.ask_prices[4])), dim=0)[0]
    curr_price[curr_price < 7.3] = 7.4
    print(curr_price)

    for env_i in range(num_envs):
        actions[:, env_i, 0] = curr_price - prev_price
        actions[:, env_i, 1] = volume_smooths - 0.75
    actions[:, :, 1] = actions[:, :, 1].clip(-1, +1)

    plt.plot(xs, curr_price, color='orange', label='VWAP price', linestyle='-')  # todo 用橙色把 vmap策略的 执行价格画出来

    plt.title(f'ask bid price (from level 1 to 5)')
    plt.legend()
    plt.grid()
    plt.show()

    # '''policy in env'''
    # ary_remain_quantity = torch.zeros((num_envs, env.max_len + 1), dtype=torch.float32, device=env.device)
    # ary_self_quantity = torch.zeros((num_envs, env.max_len + 1), dtype=torch.float32, device=env.device)
    #
    # cumulative_rewards = torch.zeros(num_envs, dtype=torch.float32, device=env.device)
    # for i in range(1, env.max_len + 1):
    #     action = actions[i]
    #     state, reward, done, _ = env.step(action)
    #     cumulative_rewards += reward
    #     if done[0]:
    #         break
    #
    #     ary_remain_quantity[:, i] = env.remain_quantity
    #     ary_self_quantity[:, i] = env.quantity
    #
    # ary_delta_quantity = ary_remain_quantity.clone()
    # ary_delta_quantity[:, 1:] -= ary_delta_quantity[:, :-1]
    # ary_delta_quantity = ary_delta_quantity[0]
    #
    # k = 5
    # smooths = ary_delta_quantity.clone()
    # for i in range(1, k):
    #     smooths[i:] += ary_delta_quantity[:-i]
    #     smooths[:-i] += ary_delta_quantity[i:]
    # smooths /= 2 * k - 1  # convolve
    # smooths[:k] = smooths[k]
    # smooths[-k:] = smooths[-k]
    #
    # smooths = ary_delta_quantity.cpu().data.numpy()
    #
    # plt.plot(xs, smooths, label='VWAP quantity', linestyle='-')
    #
    # plt.title(f'ask bid price (from level 1 to 5)')
    # plt.legend()
    # plt.grid()
    # plt.show()


def check__ask_price_volume_with_star():
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # OMP: Error #15: Initializing libiomp5md.dll

    import matplotlib.pyplot as plt
    import numpy as np

    num_envs = 2
    smooth_kernel = 7
    share_name = ['000768_XSHE', '000685_XSHE'][1]
    env = OrderExecutionVecEnvForEval(num_envs=num_envs,
                                      beg_date='2022-09-09',
                                      end_date='2022-09-09',
                                      share_name=share_name)
    env.if_random = False
    env.reset()

    max_len1 = env.max_len + 1  # after env.reset()
    xs = np.arange(max_len1)
    print('xs.shape', xs.shape)

    '''ask bid price (from level 1 to 5)'''
    from matplotlib.cm import get_cmap
    color_map = get_cmap('bwr')  # Blue White Red, input 0.0 ~ 1.0 or 0 ~ 1000

    ask_prices = np.array(env.ask_prices)
    print('ask_prices.shape', ask_prices.shape)
    n_level, max_len1 = ask_prices.shape
    for i in range(n_level):  # todo 这里的代码，把 askPrices 画出来
        face_color = color_map(float(1 - i / n_level) * 0.2 + 0.2)  # todo 使用蓝色渐变
        if i + 1 == n_level:
            plot_ask_price = np.zeros_like(ask_prices[i]) + np.nanmax(ask_prices[i])
            plt.fill_between(xs, ask_prices[i], plot_ask_price, facecolor=face_color)
        else:
            plt.fill_between(xs, ask_prices[i], ask_prices[i + 1], facecolor=face_color)

    bid_prices = np.array(env.bid_prices)
    print('bid_prices.shape', bid_prices.shape)
    n_level, max_len1 = bid_prices.shape
    for i in range(n_level):  # todo 这里的代码，把 bidPrices 画出来
        # face_color = color_map(float(i / n_level) * 0.3 + 0.5 + 0.1)  # red  # todo 使用红色渐变
        face_color = color_map(float(1 - i / n_level) * 0.2 + 0.2)  # blue  # todo 使用蓝色渐变
        if i + 1 == n_level:
            plot_bid_price = np.zeros_like(bid_prices[i]) + np.nanmin(bid_prices[i])
            plt.fill_between(xs, bid_prices[i], plot_bid_price, facecolor=face_color)
        else:
            plt.fill_between(xs, bid_prices[i], bid_prices[i + 1], facecolor=face_color)

    last_price = np.array(env.last_price)
    plt.plot(xs, last_price, color='blue', label='last price')  # todo 用蓝色把 last price 画出来

    '''policy action'''
    actions = torch.zeros((max_len1, num_envs, 2), dtype=torch.float32, device=env.device)
    print('actions.shape', actions.shape)
    # 0: the delta price is 0 in default
    # 1: the quantity scale is +1 in default

    '''policy: TWAP (one times of basic_quantity)'''
    # actions[:, :, 0] = 0.0
    # actions[:, :, 1] = 0.0  # (0.0+1) times of basic_quantity

    '''policy: VWAP (using the data in future)'''
    volume_weights = (env.volume / env.volume.mean() - 1) / env.volume.std(dim=0) + 1
    volume_smooths = torch_convolve(volume_weights, k=smooth_kernel, dim=0)

    prev_price = env.last_price.clone()
    prev_price[1:] = env.last_price[:-1]
    curr_price = env.last_price * ((volume_smooths - 1.0) * 2 * env.last_price.mean() + 1.0)
    curr_price = torch.round(curr_price * 100) / 100
    curr_price = torch.min(torch.stack((curr_price, env.ask_prices[4])), dim=0)[0]

    for env_i in range(num_envs):
        actions[:, env_i, 0] = curr_price - prev_price

        action_quantity = (volume_smooths - volume_smooths.mean()) * 12e3 + 1.8
        actions[:, env_i, 1] = action_quantity - 1
    actions[:, :, 1] = actions[:, :, 1].clip(-1, +1 + 3)

    '''policy in env'''
    env_i = 0
    ten_remain_quantity = torch.zeros((num_envs, env.max_len + 1), dtype=torch.float32, device=env.device)
    ten_remain_quantity[:, 0] = env.remain_quantity
    ten_sell_quantity = torch.zeros((num_envs, env.max_len + 1), dtype=torch.float32, device=env.device)
    ten_sell_quantity[:, 0] = env.get_curr_quantity(actions[0][:, 1])
    ten_curr_price = torch.zeros((num_envs, env.max_len + 1), dtype=torch.float32, device=env.device)
    ten_curr_price[:, 0] = env.get_curr_price(actions[0][:, 0])

    ten_rewards = torch.zeros((num_envs, env.max_len + 1), dtype=torch.float32, device=env.device)
    ten_rewards[:, 0] = 0

    for i in range(1, env.max_len + 1):
        action = actions[i]
        state, reward, done, _ = env.step(action)
        ten_rewards[:, i] = reward
        if done[0]:
            break

        ten_remain_quantity[:, i] = env.remain_quantity
        ten_sell_quantity[:, i] = env.curr_quantity
        ten_curr_price[:, i] = env.curr_price

    # ary_remain_quantity = ten_remain_quantity[env_i].cpu().data.numpy()
    # plt.plot(xs, ary_remain_quantity, label='VWAP remain_quantity', linestyle='-')

    ten_exec_quantity = torch.zeros_like(ten_remain_quantity)
    ten_exec_quantity[:, 1:] = ten_remain_quantity[:, :-1] - ten_remain_quantity[:, 1:]

    filled_bool = (ten_exec_quantity == ten_sell_quantity)[env_i]
    not_filled_bool = (ten_exec_quantity < ten_sell_quantity)[env_i]

    """
    plt.scatter(marker=(5, 1)) # marker=(5, 1)， 表示5角星里的第1款
    https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/scatter_star_poly.html
    """
    # plt.plot(xs, curr_price, color='orange', label='VWAP price', linestyle='-')  # todo 用橙色把 vmap策略的 执行价格画出来
    filled_xs = xs[filled_bool]
    filled_price = curr_price[filled_bool]
    plt.scatter(filled_xs, filled_price, color='orange', label='VWAP price (filled)', marker=(5, 1))
    not_filled_xs = xs[not_filled_bool]
    not_filled_price = curr_price[not_filled_bool]
    plt.scatter(not_filled_xs, not_filled_price, color='brown', label='VWAP price (not filled)', marker=(5, 1))

    plt.title(f'ask bid price (from level 1 to 5)')
    plt.legend()
    plt.grid()
    plt.show()

    '''draw executed_quantity <= sell_quantity'''
    # smo_exec_quantity = torch_convolve(ten_exec_quantity.T, k=smooth_kernel, dim=0).T  # todo smooth
    # ary_exec_quantity = smo_exec_quantity[env_i].cpu().data.numpy()
    # plt.plot(xs, ary_exec_quantity, label='VWAP executed_quantity', linestyle='-')
    #
    # smo_sell_quantity = torch_convolve(ten_sell_quantity.T, k=smooth_kernel, dim=0).T  # todo smooth
    # ary_sell_quantity = smo_sell_quantity.cpu().data.numpy()[env_i]
    # plt.plot(xs, ary_sell_quantity, label='VWAP sell_quantity', linestyle='-')
    #
    # plt.title(f'ask bid price (from level 1 to 5)')
    # plt.legend()
    # plt.grid()
    # plt.show()


def torch_convolve(inp, k=9, dim=0):
    assert dim == 0

    out = inp.clone()
    for i in range(1, k):
        out[i:] += inp[:-i]
        out[:-i] += inp[i:]
    out /= 2 * k - 1  # convolve
    out[:k] = out[k]
    out[-k:] = out[-k]
    return out


if __name__ == '__main__':
    # check__ask_price_volume()
    check__ask_price_volume_with_star()
