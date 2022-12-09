import sys
import gym
from elegantrl.run import train_agent, train_agent_multiprocessing
from elegantrl.config import Config, get_gym_env_args, build_env
from elegantrl.agent import AgentPPO


def train_ppo_a2c_for_pendulum():
    from elegantrl.envs.CustomGymEnv import PendulumEnv

    agent_class = AgentPPO  # DRL algorithm name
    env_class = PendulumEnv  # run a custom env: PendulumEnv, which based on OpenAI pendulum
    env_args = {
        'env_name': 'Pendulum',  # Apply torque on the free end to swing a pendulum into an upright position
        'max_step': 200,  # the max step number of an episode.
        'state_dim': 3,  # the x-y coordinates of the pendulum's free end and its angular velocity.
        'action_dim': 1,  # the torque applied to free end of the pendulum
        'if_discrete': False  # continuous action space, symbols → direction, value → force
    }
    get_gym_env_args(env=PendulumEnv(), if_print=True)  # return env_args

    args = Config(agent_class, env_class, env_args)  # see `config.py Arguments()` for hyperparameter explanation
    args.break_step = int(8e4)  # break training if 'total_step > break_step'
    args.net_dims = (128, 64)  # the middle layer dimension of MultiLayer Perceptron
    args.gamma = 0.97  # discount factor of future rewards
    args.horizon_len = args.max_step * 4

    args.repeat_times = 32  # repeatedly update network using ReplayBuffer to keep critic's loss small
    args.learning_rate = 2e-4
    args.state_value_tau = 0.1  # the tau of normalize for value and state `std = (1-std)*std + tau*std`

    args.gpu_id = GPU_ID
    args.num_workers = 4
    if_check = False
    if if_check:
        train_agent(args)
    else:
        train_agent_multiprocessing(args)
    """
    -2000 < -1200 < -200 < -80
    ################################################################################
    ID     Step    Time |    avgR   stdR   avgS  stdS |    expR   objC   etc.
    0  6.40e+03      14 |-1192.19  199.4    200     0 |   -1.44  32.65   0.02   0.01
    0  6.40e+03      14 |-1192.19
    0  2.88e+04      38 | -952.89   70.4    200     0 |   -1.39  13.91   0.02  -0.03
    0  2.88e+04      38 | -952.89
    0  5.12e+04      65 | -421.47   72.3    200     0 |   -1.38  12.35   0.00  -0.06
    0  5.12e+04      65 | -421.47
    0  7.36e+04      91 | -168.78   74.8    200     0 |   -1.28   4.49   0.04  -0.16
    0  7.36e+04      91 | -168.78
    | TrainingTime:     103 | SavedDir: ./Pendulum_PPO_0
    """


def train_ppo_a2c_for_pendulum_vec_env():
    from elegantrl.envs.CustomGymEnv import PendulumVecEnv
    num_envs = 4

    agent_class = AgentPPO  # DRL algorithm name
    env_class = PendulumVecEnv  # run a custom env: PendulumEnv, which based on OpenAI pendulum
    env_args = {
        'env_name': 'Pendulum',  # Apply torque on the free end to swing a pendulum into an upright position
        'num_envs': num_envs,  # the number of sub envs in vectorized env
        'max_step': 200,  # the max step number in an episode for evaluation
        'state_dim': 3,  # the x-y coordinates of the pendulum's free end and its angular velocity.
        'action_dim': 1,  # the torque applied to free end of the pendulum
        'if_discrete': False  # continuous action space, symbols → direction, value → force
    }
    get_gym_env_args(env=PendulumVecEnv(num_envs=num_envs), if_print=True)  # return env_args

    args = Config(agent_class, env_class, env_args)  # see `config.py Arguments()` for hyperparameter explanation
    args.break_step = int(8e4)
    args.net_dims = (128, 64)  # the middle layer dimension of MultiLayer Perceptron
    args.gamma = 0.97  # discount factor of future rewards
    args.reward_scale = 2 ** -2

    args.horizon_len = args.max_step * 1
    args.repeat_times = 16  # repeatedly update network using ReplayBuffer to keep critic's loss small
    args.learning_rate = 4e-4
    args.state_value_tau = 0.2  # the tau of normalize for value and state `std = (1-std)*std + tau*std`

    args.gpu_id = GPU_ID
    args.num_workers = 4
    train_agent_multiprocessing(args)  # train_agent(args)
    """
    -2000 < -1200 < -200 < -80
    ################################################################################
    ID     Step    Time |    avgR   stdR   avgS  stdS |    expR   objC   etc.
    0  1.60e+03       9 |-1216.88  238.3    200     0 |   -1.42   9.23  -0.07   0.00
    0  1.60e+03       9 |-1216.88
    0  2.16e+04      39 |-1025.17   99.0    200     0 |   -1.41   3.30  -0.03   0.01
    0  2.16e+04      39 |-1025.17
    0  4.16e+04      71 | -704.75   89.1    200     0 |   -1.37   2.69   0.02  -0.04
    0  4.16e+04      71 | -704.75
    0  6.16e+04     104 |  -65.52   74.6    200     0 |   -1.24   1.08   0.39  -0.18
    0  6.16e+04     104 |  -65.52
    | TrainingTime:     137 | SavedDir: ./Pendulum_PPO_0
    """


def train_ppo_a2c_for_lunar_lander_continuous():
    agent_class = AgentPPO  # DRL algorithm name
    env_class = gym.make  # run a custom env: PendulumEnv, which based on OpenAI pendulum
    env_args = {'env_name': 'LunarLanderContinuous-v2',
                'num_envs': 1,
                'max_step': 1000,
                'state_dim': 8,
                'action_dim': 2,
                'if_discrete': False}
    get_gym_env_args(env=gym.make('LunarLanderContinuous-v2'), if_print=True)  # return env_args

    args = Config(agent_class, env_class, env_args)  # see `config.py Arguments()` for hyperparameter explanation
    args.break_step = int(4e5)  # break training if 'total_step > break_step'
    args.net_dims = (256, 128)  # the middle layer dimension of MultiLayer Perceptron
    args.batch_size = 512
    args.gamma = 0.99  # discount factor of future rewards
    args.horizon_len = args.max_step * 2
    args.repeat_times = 16  # repeatedly update network using ReplayBuffer to keep critic's loss small
    args.reward_scale = 2 ** -1
    args.learning_rate = 2e-4
    args.state_value_tau = 0.1  # the tau of normalize for value and state `std = (1-std)*std + tau*std`
    args.lambda_gae_adv = 0.97
    args.lambda_entropy = 0.04

    args.eval_times = 32
    args.eval_per_step = 5e4

    args.gpu_id = GPU_ID
    args.num_workers = 4
    train_agent_multiprocessing(args)  # train_agent(args)
    """
    -1500 < -200 < 200 < 290
    ################################################################################
    ID     Step    Time |    avgR   stdR   avgS  stdS |    expR   objC   etc.                  
    0  1.60e+04      20 | -236.43  103.0    118    41 |   -2.87  12.21   0.12   0.02
    0  1.60e+04      20 | -236.43                                                              
    0  7.20e+04      65 | -190.03   69.5    245   200 |   -2.94   6.26   0.14   0.05
    0  7.20e+04      65 | -190.03                                                              
    0  1.28e+05     144 |  105.61  123.7    591   125 |   -3.01   2.38   0.13   0.08
    0  1.28e+05     144 |  105.61                                                              
    0  1.84e+05     235 |  124.07   78.8    866    85 |   -3.02   1.54   0.24   0.10
    0  1.84e+05     235 |  124.07                                                              
    0  2.80e+05     265 |  183.62   95.3    431    70 |   -3.11   0.86   0.09   0.13
    0  2.80e+05     265 |  183.62                                                              
    0  3.36e+05     304 |  191.30   88.5    336    85 |   -3.14   0.57   0.24   0.16
    0  3.36e+05     304 |  191.30                                                              
    0  3.92e+05     346 |  265.66   16.1    254    46 |   -3.18   0.24   0.09   0.18
    0  3.92e+05     346 |  265.66                                                              
    | TrainingTime:     350 | SavedDir: ./LunarLanderContinuous-v2_PPO_0
    """


def train_ppo_a2c_for_lunar_lander_continuous_vec_env():
    num_envs = 4
    from elegantrl.envs.CustomGymEnv import GymVecEnv

    agent_class = AgentPPO  # DRL algorithm name
    env_class = GymVecEnv  # run a custom env: PendulumEnv, which based on OpenAI pendulum
    env_args = {'env_name': 'LunarLanderContinuous-v2',
                'num_envs': num_envs,
                'max_step': 1000,
                'state_dim': 8,
                'action_dim': 2,
                'if_discrete': False}
    get_gym_env_args(env=GymVecEnv('LunarLanderContinuous-v2', num_envs), if_print=True)  # return env_args

    args = Config(agent_class, env_class, env_args)  # see `config.py Arguments()` for hyperparameter explanation
    args.break_step = int(2e5)  # break training if 'total_step > break_step'
    args.net_dims = (256, 128, 64)  # the middle layer dimension of MultiLayer Perceptron
    args.batch_size = 512
    args.gamma = 0.99  # discount factor of future rewards
    args.horizon_len = args.max_step
    args.repeat_times = 64  # repeatedly update network using ReplayBuffer to keep critic's loss small
    args.reward_scale = 2 ** -1
    args.learning_rate = 2e-4
    args.state_value_tau = 0.1  # the tau of normalize for value and state `std = (1-std)*std + tau*std`
    args.lambda_gae_adv = 0.97
    args.lambda_entropy = 0.04

    args.eval_times = 32
    args.eval_per_step = 2e4

    args.gpu_id = GPU_ID
    args.num_workers = 4
    train_agent_multiprocessing(args)  # train_agent(args)
    """
    -1500 < -200 < 200 < 290
    ################################################################################
    ID     Step    Time |    avgR   stdR   avgS  stdS |    expR   objC   etc.
    0  8.00e+03      35 | -138.30   45.6     70    13 |   -2.86   8.82   0.12   0.02
    0  8.00e+03      35 | -138.30
    0  2.80e+04      72 |  -56.35  102.9    293    81 |   -2.94   2.94   0.13   0.05
    0  2.80e+04      72 |  -56.35
    0  4.80e+04     120 |  145.22  111.2    572    78 |   -2.97   2.04   0.24   0.07
    0  4.80e+04     120 |  145.22
    0  7.20e+04     151 |  226.12   77.3    314    36 |   -2.97   0.79   0.27   0.07
    0  7.20e+04     151 |  226.12
    0  9.20e+04     181 |  257.79   28.7    284    53 |   -3.00   0.68   0.33   0.08
    0  9.20e+04     181 |  257.79
    0  1.12e+05     210 |  269.37   24.5    236    24 |   -3.01   0.44   0.36   0.09
    0  1.12e+05     210 |  269.37
    0  1.32e+05     240 |  275.08   21.4    223    33 |   -3.03   0.71   0.48   0.10
    0  1.32e+05     240 |  275.08
    0  1.52e+05     270 |  276.60   18.6    209    22 |   -3.09   0.51   0.37   0.13
    0  1.52e+05     270 |  276.60
    0  1.72e+05     300 |  285.19   20.1    197    17 |   -3.13   0.34   0.44   0.15
    0  1.72e+05     300 |  285.19
    0  1.92e+05     331 |  278.00   20.8    221    16 |   -3.16   0.08   0.09   0.17
    | TrainingTime:     332 | SavedDir: ./LunarLanderContinuous-v2_PPO_0
    """


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


def train_ppo_a2c_for_stock_trading():
    from elegantrl.envs.StockTradingEnv import StockTradingEnv
    id0 = 0
    id1 = int(1113 * 0.8)
    id2 = 1113
    gamma = 0.99

    agent_class = AgentPPO
    env_class = StockTradingEnv
    env_args = {'env_name': 'StockTradingEnv-v2',
                'num_envs': 1,
                'max_step': id2 - id1 - 1,
                'state_dim': 151,
                'action_dim': 15,
                'if_discrete': False,

                'gamma': gamma,
                'beg_idx': id0,
                'end_idx': id1, }
    # get_gym_env_args(env=StockTradingEnv(), if_print=True)  # return env_args

    args = Config(agent_class, env_class, env_args)  # see `config.py Arguments()` for hyperparameter explanation
    args.break_step = int(2e5)  # break training if 'total_step > break_step'
    args.net_dims = (128, 64)  # the middle layer dimension of MultiLayer Perceptron
    args.gamma = gamma  # discount factor of future rewards
    args.horizon_len = args.max_step

    args.repeat_times = 16  # repeatedly update network using ReplayBuffer to keep critic's loss small
    args.learning_rate = 1e-4
    args.state_value_tau = 0.1  # the tau of normalize for value and state `std = (1-std)*std + tau*std`

    args.eval_times = 2 ** 5
    args.eval_per_step = int(2e4)
    args.eval_env_class = StockTradingEnv
    args.eval_env_args = {'env_name': 'StockTradingEnv-v2',
                          'num_envs': 1,
                          'max_step': id2 - id1 - 1,
                          'state_dim': 151,
                          'action_dim': 15,
                          'if_discrete': False,

                          'beg_idx': id1,
                          'end_idx': id2, }

    args.gpu_id = GPU_ID
    args.num_workers = 4
    train_agent_multiprocessing(args)  # train_agent(args)
    """
    RewardRange: 0.0 < 1.0 < 1.5 < 2.0
    ################################################################################
    ID     Step    Time |    avgR   stdR   avgS  stdS |    expR   objC   etc.
    0  7.12e+03       8 |    1.08    0.1    222     0 |  -21.40   4.36   0.23   0.00
    0  7.12e+03       8 |    1.08
    0  2.85e+04      21 |    1.64    0.1    222     0 |  -21.36   6.70   0.22   0.01
    0  2.85e+04      21 |    1.64
    0  4.98e+04      34 |    1.58    0.1    222     0 |  -21.47   4.98   0.22   0.01
    0  7.12e+04      47 |    1.53    0.1    222     0 |  -21.47   3.99   0.24   0.01
    0  9.26e+04      60 |    1.52    0.1    222     0 |  -21.55   3.80   0.25   0.02
    0  1.14e+05      73 |    1.51    0.1    222     0 |  -21.61   3.16   0.26   0.02
    0  1.35e+05      86 |    1.53    0.1    222     0 |  -21.63   3.48   0.18   0.02
    0  1.57e+05     100 |    1.50    0.1    222     0 |  -21.67   2.68   0.22   0.02
    0  1.78e+05     114 |    1.51    0.1    222     0 |  -21.80   2.18   0.22   0.03
    0  1.99e+05     129 |    1.50    0.1    222     0 |  -21.76   2.10   0.24   0.03
    | TrainingTime:     130 | SavedDir: ./StockTradingEnv-v2_PPO_0
    """


def train_ppo_a2c_for_stock_trading_vec_env():
    from elegantrl.envs.StockTradingEnv import StockTradingVecEnv
    id0 = 0
    id1 = int(1113 * 0.8)
    id2 = 1113
    num_envs = 2 ** 11
    gamma = 0.99

    agent_class = AgentPPO
    env_class = StockTradingVecEnv
    env_args = {'env_name': 'StockTradingVecEnv-v2',
                'num_envs': num_envs,
                'max_step': id2 - id1 - 1,
                'state_dim': 151,
                'action_dim': 15,
                'if_discrete': False,

                'gamma': gamma,
                'beg_idx': id0,
                'end_idx': id1, }
    # get_gym_env_args(env=StockTradingVecEnv(), if_print=True)  # return env_args

    args = Config(agent_class, env_class, env_args)  # see `config.py Arguments()` for hyperparameter explanation
    args.break_step = int(1e5)  # break training if 'total_step > break_step'
    args.net_dims = (128, 64)  # the middle layer dimension of MultiLayer Perceptron
    args.gamma = gamma  # discount factor of future rewards
    args.horizon_len = args.max_step

    args.repeat_times = 16  # repeatedly update network using ReplayBuffer to keep critic's loss small
    args.learning_rate = 2e-4
    args.state_value_tau = 0.1  # the tau of normalize for value and state `std = (1-std)*std + tau*std`

    args.eval_times = 2 ** 14
    args.eval_per_step = int(2e4)
    args.eval_env_class = StockTradingVecEnv
    args.eval_env_args = {'env_name': 'StockTradingVecEnv-v2',
                          'num_envs': num_envs,
                          'max_step': id2 - id1 - 1,
                          'state_dim': 151,
                          'action_dim': 15,
                          'if_discrete': False,

                          'beg_idx': id1,
                          'end_idx': id2, }

    args.gpu_id = GPU_ID
    args.random_seed = GPU_ID
    args.num_workers = 2
    train_agent_multiprocessing(args)  # train_agent(args)
    """
    0.0 < 1.0 < 1.5 < 2.0
    ################################################################################
    ID     Step    Time |    avgR   stdR   avgS  stdS |    expR   objC   etc.
    0  8.88e+02      30 |    1.52    0.2    222     0 |  -21.29  19.51   0.19   0.00
    0  8.88e+02      30 |    1.52
    0  2.13e+04     180 |    1.52    0.2    222     0 |  -21.58   1.74   0.23   0.02
    0  4.17e+04     333 |    1.52    0.2    222     0 |  -21.85   0.81   0.24   0.04
    0  6.22e+04     485 |    1.52    0.2    222     0 |  -22.16   0.56   0.24   0.06
    0  8.26e+04     635 |    1.52    0.2    222     0 |  -22.45   0.50   0.21   0.08
    | TrainingTime:     746 | SavedDir: ./StockTradingVecEnv-v2_PPO_0
    """


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
    # DRL_ID = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    # ENV_ID = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    # train_ppo_a2c_for_bipedal_walker_vec_env()
    # train_ppo_a2c_for_stock_trading_vec_env()
    train_ppo_a2c_for_order_execution_vec_env()
