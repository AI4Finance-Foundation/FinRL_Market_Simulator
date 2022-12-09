import sys
import gym
from elegantrl.run import train_agent, train_agent_multiprocessing
from elegantrl.config import Config, get_gym_env_args, build_env
from elegantrl.agent import AgentPPO


def train_ppo_a2c_for_bipedal_walker():
    agent_class = AgentPPO  # DRL algorithm name
    env_class = gym.make  # run a custom env: PendulumEnv, which based on OpenAI pendulum
    env_args = {'env_name': 'BipedalWalker-v3',
                'num_envs': 1,
                'max_step': 1600,
                'state_dim': 24,
                'action_dim': 4,
                'if_discrete': False}
    get_gym_env_args(env=gym.make('BipedalWalker-v3'), if_print=True)  # return env_args

    args = Config(agent_class, env_class, env_args)  # see `config.py Arguments()` for hyperparameter explanation
    args.break_step = int(8e5)  # break training if 'total_step > break_step'
    args.net_dims = (256, 128, 128)  # the middle layer dimension of MultiLayer Perceptron
    args.batch_size = 512
    args.gamma = 0.97  # discount factor of future rewards
    args.horizon_len = args.max_step * 3
    args.repeat_times = 32  # repeatedly update network using ReplayBuffer to keep critic's loss small
    args.learning_rate = 1e-4
    args.state_value_tau = 0.01  # the tau of normalize for value and state `std = (1-std)*std + tau*std`
    args.lambda_gae_adv = 0.93
    args.lambda_entropy = 0.02
    args.clip_ratio = 0.4

    args.eval_times = 16
    args.eval_per_step = 8e4
    args.if_keep_save = False  # keeping save the checkpoint. False means save until stop training.

    args.gpu_id = GPU_ID
    args.random_seed = GPU_ID
    args.num_workers = 2
    train_agent_multiprocessing(args)  # train_agent(args)
    """
    -200 < -150 < 300 < 330
    ################################################################################
    ID     Step    Time |    avgR   stdR   avgS  stdS |    expR   objC   etc.
    0  1.92e+04      27 | -105.34    5.9    158    44 |   -5.71   0.49   0.65   0.01
    0  1.92e+04      27 | -105.34
    0  1.06e+05     131 |  -63.90    1.9   1600     0 |   -5.96   0.37   0.13   0.07
    0  1.06e+05     131 |  -63.90
    0  1.92e+05     223 |   57.48    3.5   1600     0 |   -5.90   0.03   0.13   0.05
    0  1.92e+05     223 |   57.48
    0  2.78e+05     308 |  104.86  117.1   1211   493 |   -5.65   0.07   0.18  -0.01
    0  2.78e+05     308 |  104.86
    0  3.65e+05     395 |  123.77  147.7    990   487 |   -5.63   0.27   0.13  -0.01
    0  3.65e+05     395 |  123.77
    0  4.51e+05     486 |  236.73  130.2   1038   361 |   -5.65   0.23   0.14  -0.00
    0  4.51e+05     486 |  236.73
    0  5.38e+05     575 |  286.09    1.4   1059    20 |   -5.72   0.17   0.14   0.01
    0  5.38e+05     575 |  286.09
    0  6.24e+05     664 |  276.44   44.7   1010    53 |   -5.76   0.20   0.13   0.02
    0  7.10e+05     753 |  287.70    1.7    986    24 |   -5.84   0.13   0.12   0.04
    0  7.10e+05     753 |  287.70
    0  7.97e+05     843 |  223.00  119.7    812   232 |   -5.95   0.12   0.14   0.07
    | TrainingTime:     845 | SavedDir: ./BipedalWalker-v3_PPO_2
    """


def train_ppo_a2c_for_bipedal_walker_vec_env():
    env_name = 'BipedalWalker-v3'
    num_envs = 4
    from elegantrl.envs.CustomGymEnv import GymVecEnv

    agent_class = AgentPPO  # DRL algorithm name
    env_class = GymVecEnv  # run a custom env: PendulumEnv, which based on OpenAI pendulum
    env_args = {'env_name': env_name,
                'num_envs': num_envs,
                'max_step': 1600,
                'state_dim': 24,
                'action_dim': 4,
                'if_discrete': False}
    get_gym_env_args(env=build_env(env_class, env_args), if_print=True)  # return env_args

    args = Config(agent_class, env_class, env_args)  # see `config.py Arguments()` for hyperparameter explanation
    args.break_step = int(8e5)  # break training if 'total_step > break_step'
    args.net_dims = (256, 128, 128)  # the middle layer dimension of MultiLayer Perceptron
    args.batch_size = 512
    args.gamma = 0.98
    args.horizon_len = args.max_step // 1
    args.repeat_times = 32  # repeatedly update network using ReplayBuffer to keep critic's loss small
    args.learning_rate = 2e-4
    args.state_value_tau = 0.01  # the tau of normalize for value and state `std = (1-std)*std + tau*std`
    args.lambda_gae_adv = 0.93
    args.lambda_entropy = 0.02

    args.eval_times = 16
    args.eval_per_step = 5e4
    args.if_keep_save = False  # keeping save the checkpoint. False means save until stop training.

    args.gpu_id = GPU_ID
    args.random_seed = GPU_ID
    args.num_workers = 2

    if_check = False
    if if_check:
        train_agent_multiprocessing(args)
    else:
        train_agent(args)
    """
    -200 < -150 < 300 < 330
    ################################################################################
    ID     Step    Time |    avgR   stdR   avgS  stdS |    expR   objC   etc.
    0  6.40e+03      33 | -107.05    5.9    169    30 |   -5.67   1.30   0.69  -0.01
    0  6.40e+03      33 | -107.05
    0  5.76e+04     113 |  -37.95    2.0   1600     0 |   -5.70   0.05   0.12  -0.00
    0  5.76e+04     113 |  -37.95
    0  1.09e+05     196 |  163.69   76.5   1497   287 |   -5.39   0.07   0.24  -0.08
    0  1.09e+05     196 |  163.69
    0  1.60e+05     280 |   28.24  120.4    690   434 |   -5.33   0.46   0.17  -0.08
    0  2.11e+05     364 |   97.72  147.8    801   396 |   -5.32   0.28   0.18  -0.09
    0  2.62e+05     447 |  254.85   78.5   1071   165 |   -5.37   0.29   0.16  -0.08
    0  2.62e+05     447 |  254.85
    0  3.14e+05     530 |  274.90   61.5   1001   123 |   -5.48   0.34   0.15  -0.04
    0  3.14e+05     530 |  274.90
    0  3.65e+05     611 |  196.47  121.1    806   220 |   -5.60   0.35   0.18  -0.01
    0  4.16e+05     689 |  250.12   89.0    890   143 |   -5.78   0.32   0.18   0.03
    0  4.67e+05     768 |  282.29   25.5    909    17 |   -5.94   0.47   0.17   0.07
    0  4.67e+05     768 |  282.29
    0  5.18e+05     848 |  289.36    1.4    897    14 |   -6.07   0.26   0.16   0.10
    0  5.18e+05     848 |  289.36
    0  5.70e+05     929 |  283.14   33.8    874    35 |   -6.29   0.27   0.13   0.16
    0  6.21e+05    1007 |  288.53    1.1    870    13 |   -6.52   0.22   0.15   0.21
    0  6.72e+05    1087 |  288.50    0.9    856    13 |   -6.68   0.40   0.15   0.25
    0  7.23e+05    1167 |  286.92    1.3    842    16 |   -6.86   0.40   0.15   0.30
    0  7.74e+05    1246 |  264.75   74.0    790   122 |   -7.10   0.42   0.18   0.36
    | TrainingTime:    1278 | SavedDir: ./BipedalWalker-v3_PPO_5
    """


'''train'''


def train_ppo_a2c_for_order_execution_vec_env():
    from elegantrl.envs.OrderExecutionEnv import OrderExecutionVecEnv

    # num_envs = 2 ** 9
    total = 2 ** (9 + 12)  # todo
    if GPU_ID == 1:
        num_envs = 2 ** 7
    elif GPU_ID == 6:
        num_envs = 2 ** 9
    elif GPU_ID == 7:
        num_envs = 2 ** 11
    else:
        assert GPU_ID == 0
        num_envs = 2 ** 13

    gamma = 0.998
    n_stack = 8

    agent_class = AgentPPO
    env_class = OrderExecutionVecEnv
    env_args = {'env_name': 'OrderExecutionVecEnv-v0',
                'num_envs': num_envs,
                'max_step': 5000,
                'state_dim': 48 * n_stack,
                'action_dim': 2,
                'if_discrete': False,

                'beg_date': '2022-08-09',
                'end_date': '2022-09-09',
                'if_random': False}
    # get_gym_env_args(env=OrderExecutionVecEnv(), if_print=True)  # return env_args

    args = Config(agent_class, env_class, env_args)  # see `config.py Arguments()` for hyperparameter explanation
    args.break_step = int(4e5)  # break training if 'total_step > break_step'
    args.net_dims = (256, 128)  # the middle layer dimension of MultiLayer Perceptron
    args.gamma = gamma  # discount factor of future rewards
    args.horizon_len = total // num_envs

    args.batch_size = args.horizon_len * num_envs // 16
    args.repeat_times = 8  # repeatedly update network using ReplayBuffer to keep critic's loss small
    args.learning_rate = 1e-4
    args.state_value_tau = 0.02

    args.eval_times = num_envs * 5
    args.eval_per_step = int(8e2)
    args.eval_env_class = env_class
    # args.eval_env_args = env_args
    args.eval_env_args = {'env_name': 'OrderExecutionVecEnv-v0',
                          'num_envs': num_envs,
                          'max_step': 5000,
                          'state_dim': 48 * n_stack,
                          'action_dim': 2,
                          'if_discrete': False,

                          'beg_date': '2022-09-10',
                          'end_date': '2022-09-16',
                          'if_random': False}

    args.gpu_id = GPU_ID
    args.random_seed = GPU_ID
    args.num_workers = 1
    train_agent_multiprocessing(args)
    # train_agent(args)
    """
    0.0 < 1.0 < 1.5 < 2.0
    ################################################################################
    ID     Step    Time |    avgR   stdR   avgS  stdS |    expR   objC   etc.
    0  9.48e+03     423 |   22.63    5.7   4739     0 |   -2.83   1.89   0.00  -0.02
    0  9.48e+03     423 |   22.63
    0  3.32e+04     769 |   26.11    5.6   4739     0 |   -2.68   1.43   0.01  -0.10 
    0  3.32e+04     769 |   26.11  
    0  5.69e+04    1147 |   26.66    5.0   4739     0 |   -2.55   1.13   0.03  -0.14
    0  5.69e+04    1147 |   26.66   
    0  8.06e+04    1591 |   26.81    5.8   4739     0 |   -2.51   0.57   0.00  -0.17
    0  8.06e+04    1591 |   26.81  
    0  1.04e+05    1977 |   26.71    5.5   4739     0 |   -2.35   0.93   0.00  -0.24
    """


if __name__ == '__main__':
    GPU_ID = int(sys.argv[1]) if len(sys.argv) > 1 else 0  # >=0 means GPU ID, -1 means CPU

    train_ppo_a2c_for_order_execution_vec_env()
