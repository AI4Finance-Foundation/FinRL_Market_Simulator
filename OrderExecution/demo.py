import sys
import gym
from elegantrl.run import train_agent, train_agent_multiprocessing
from elegantrl.config import Config, get_gym_env_args, build_env
from elegantrl.agent import AgentPPO

'''train'''


def train_ppo_a2c_for_order_execution_vec_env():
    from OrderExecutionEnv import OrderExecutionVecEnv
    num_envs = 2 ** 9

    gamma = 0.999
    n_stack = 8

    agent_class = AgentPPO
    env_class = OrderExecutionVecEnv
    env_args = {'env_name': 'OrderExecutionVecEnv-v2',
                'num_envs': num_envs,
                'max_step': 5000,
                'state_dim': 48 * n_stack,
                'action_dim': 2,
                'if_discrete': False,

                'share_name': '000768_XSHE',
                'beg_date': '2022-06-09',
                'end_date': '2022-09-09',
                'if_random': False}
    if not env_args:
        get_gym_env_args(env=OrderExecutionVecEnv(), if_print=True)  # return env_args

    args = Config(agent_class, env_class, env_args)  # see `config.py Arguments()` for hyperparameter explanation
    args.break_step = int(1e6)  # break training if 'total_step > break_step'
    args.net_dims = (256, 128, 64)  # the middle layer dimension of MultiLayer Perceptron
    args.gamma = gamma  # discount factor of future rewards
    args.horizon_len = 2 ** 9

    args.batch_size = args.horizon_len * num_envs // 32
    args.repeat_times = 4  # repeatedly update network using ReplayBuffer to keep critic's loss small
    args.learning_rate = 1e-4
    args.state_value_tau = 0.01

    eval_num_envs = 16
    args.save_gap = int(8)
    args.if_keep_save = True
    args.if_over_write = False
    args.eval_per_step = int(4e3)
    args.eval_times = eval_num_envs
    from OrderExecutionEnv import OrderExecutionVecEnvForEval
    args.eval_env_class = OrderExecutionVecEnvForEval
    args.eval_env_args = env_args.copy()
    args.eval_env_args['num_envs'] = eval_num_envs
    args.eval_env_args['max_step'] = 4000 * 22
    args.eval_env_args['beg_date'] = '2022-09-10'
    args.eval_env_args['end_date'] = '2022-10-10'

    args.gpu_id = GPU_ID
    args.eval_gpu_id = GPU_ID
    args.random_seed = GPU_ID
    args.num_workers = 2

    if_check = False
    if if_check:
        train_agent(args)
    else:
        train_agent_multiprocessing(args)
    """
    0% < 100% < 120%
    ################################################################################
    ID     Step    Time |    avgR   stdR   avgS  stdS |    expR   objC   etc.
    6  1.64e+04     559 |  100.75    0.3  88000     0 |   -2.81   0.44  -0.03  -0.03
    6  1.64e+04     559 |  100.75
    6  1.64e+05    1025 |  101.19    0.5  88000     0 |   -2.58   0.37  -0.10  -0.13
    6  1.64e+05    1025 |  101.19
    6  1.72e+05    1471 |  101.21    0.5  88000     0 |   -2.58   0.50   0.01  -0.12
    6  1.72e+05    1471 |  101.21
    6  1.80e+05    1916 |  101.20    0.5  88000     0 |   -2.60   0.27  -0.14  -0.11
    6  1.88e+05    2362 |  101.21    0.5  88000     0 |   -2.63   0.63  -0.19  -0.10
    6  1.88e+05    2362 |  101.21
    6  1.97e+05    2807 |  101.22    0.5  88000     0 |   -2.64   0.58  -0.18  -0.10
    6  1.97e+05    2807 |  101.22
    6  2.05e+05    3253 |  101.24    0.5  88000     0 |   -2.64   0.25   0.04  -0.09
    6  2.05e+05    3253 |  101.24
    6  2.13e+05    3698 |  101.24    0.5  88000     0 |   -2.67   0.46  -0.05  -0.08
    6  2.13e+05    3698 |  101.24
    6  2.21e+05    4143 |  101.25    0.5  88000     0 |   -2.68   0.33  -0.01  -0.07
    6  2.21e+05    4143 |  101.25
    6  2.29e+05    4589 |  101.26    0.5  88000     0 |   -2.69   0.50   0.08  -0.06
    6  2.29e+05    4589 |  101.26
    6  2.38e+05    5034 |  101.27    0.5  88000     0 |   -2.71   0.26   0.05  -0.05
    6  2.38e+05    5034 |  101.27
    """


'''help users understand Vectorized env by comparing with single env'''


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


if __name__ == '__main__':
    GPU_ID = int(sys.argv[1]) if len(sys.argv) > 1 else 0  # >=0 means GPU ID, -1 means CPU

    train_ppo_a2c_for_order_execution_vec_env()
