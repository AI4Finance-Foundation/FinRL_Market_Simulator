import os
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from elegantrl.config import Config

'''net'''


class ActorBase(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.net = None  # build_mlp(dims=[state_dim, *dims, action_dim])
        self.explore_noise_std = None  # standard deviation of exploration action noise
        self.ActionDist = torch.distributions.normal.Normal

        self.state_avg = nn.Parameter(torch.zeros((state_dim,)), requires_grad=False)
        self.state_std = nn.Parameter(torch.ones((state_dim,)), requires_grad=False)

    def state_norm(self, state: Tensor) -> Tensor:
        return (state - self.state_avg) / self.state_std


class ActorPPO(ActorBase):
    def __init__(self, dims: [int], state_dim: int, action_dim: int):
        super().__init__(state_dim=state_dim, action_dim=action_dim)
        self.net = build_mlp(dims=[state_dim, *dims, action_dim])
        layer_init_with_orthogonal(self.net[-1], std=0.1)

        self.action_std_log = nn.Parameter(torch.zeros((1, action_dim)), requires_grad=True)  # trainable parameter

    def forward(self, state: Tensor) -> Tensor:
        state = self.state_norm(state)
        return self.net(state).tanh()  # action.tanh()

    def get_action(self, state: Tensor) -> (Tensor, Tensor):  # for exploration
        state = self.state_norm(state)
        action_avg = self.net(state)
        action_std = self.action_std_log.exp()

        dist = self.ActionDist(action_avg, action_std)
        action = dist.sample()
        logprob = dist.log_prob(action).sum(1)
        return action, logprob

    def get_logprob_entropy(self, state: Tensor, action: Tensor) -> (Tensor, Tensor):
        state = self.state_norm(state)
        action_avg = self.net(state)
        action_std = self.action_std_log.exp()

        dist = self.ActionDist(action_avg, action_std)
        logprob = dist.log_prob(action).sum(1)
        entropy = dist.entropy().sum(1)
        return logprob, entropy

    @staticmethod
    def convert_action_for_env(action: Tensor) -> Tensor:
        return action.tanh()


class CriticBase(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.net = None  # build_mlp(dims=[state_dim + action_dim, *dims, 1])

        self.state_avg = nn.Parameter(torch.zeros((state_dim,)), requires_grad=False)
        self.state_vam = nn.Parameter(torch.ones((state_dim,)), requires_grad=False)  # var.mean
        self.state_std = nn.Parameter(torch.ones((state_dim,)), requires_grad=False)  # sqrt(var.mean - avg.pow2)

        self.value_avg = nn.Parameter(torch.zeros((1,)), requires_grad=False)
        self.value_vam = nn.Parameter(torch.ones((1,)), requires_grad=False)  # var.mean
        self.value_std = nn.Parameter(torch.ones((1,)), requires_grad=False)  # sqrt(var.mean - avg.pow2)

    def state_norm(self, state: Tensor) -> Tensor:
        return (state - self.state_avg) / self.state_std

    def value_re_norm(self, value: Tensor) -> Tensor:
        return value * self.value_std + self.value_avg


class CriticPPO(CriticBase):
    def __init__(self, dims: [int], state_dim: int, action_dim: int):
        super().__init__(state_dim=state_dim, action_dim=action_dim)
        self.net = build_mlp(dims=[state_dim, *dims, 1])
        layer_init_with_orthogonal(self.net[-1], std=0.5)

    def forward(self, state: Tensor) -> Tensor:
        state = self.state_norm(state)
        value = self.net(state)
        value = self.value_re_norm(value)
        return value  # value


"""utils"""


def build_mlp(dims: [int], activation: nn = None, if_raw_out: bool = True) -> nn.Sequential:
    """
    build MLP (MultiLayer Perceptron)

    dims: the middle dimension, `dims[-1]` is the output dimension of this network
    activation: the activation function
    if_remove_out_layer: if remove the activation function of the output layer.
    """
    if activation is None:
        activation = nn.ReLU
    net_list = []
    for i in range(len(dims) - 1):
        net_list.extend([nn.Linear(dims[i], dims[i + 1]), activation()])
    if if_raw_out:
        del net_list[-1]  # delete the activation function of the output layer to keep raw output
    return nn.Sequential(*net_list)


def layer_init_with_orthogonal(layer, std=1.0, bias_const=1e-6):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)


'''agent'''


class AgentBase:
    """
    The basic agent of ElegantRL

    net_dims: the middle layer dimension of MLP (MultiLayer Perceptron)
    state_dim: the dimension of state (the number of state vector)
    action_dim: the dimension of action (or the number of discrete action)
    gpu_id: the gpu_id of the training device. Use CPU when cuda is not available.
    args: the arguments for agent training. `args = Config()`
    """

    def __init__(self, net_dims: [int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        self.gamma = args.gamma  # discount factor of future rewards
        self.num_envs = args.num_envs  # the number of sub envs in vectorized env. `num_envs=1` in single env.
        self.batch_size = args.batch_size  # num of transitions sampled from replay buffer.
        self.if_discrete = args.if_discrete  # if the action space of env is discrete or continuous
        self.repeat_times = args.repeat_times  # repeatedly update network using ReplayBuffer
        self.reward_scale = args.reward_scale  # an approximate target reward usually be closed to 256
        self.learning_rate = args.learning_rate  # the learning rate for network updating
        self.if_off_policy = args.if_off_policy  # whether off-policy or on-policy of DRL algorithm
        self.clip_grad_norm = args.clip_grad_norm  # clip the gradient after normalization
        self.soft_update_tau = args.soft_update_tau  # the tau of soft target update `net = (1-tau)*net + net1`
        self.state_value_tau = args.state_value_tau  # the tau of normalize for value and state

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.last_state = None  # last state of the trajectory for training. last_state.shape == (num_envs, state_dim)
        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")

        '''network'''
        act_class = getattr(self, "act_class", None)
        cri_class = getattr(self, "cri_class", None)
        self.act = self.act_target = act_class(net_dims, state_dim, action_dim).to(self.device)
        self.cri = self.cri_target = cri_class(net_dims, state_dim, action_dim).to(self.device) \
            if cri_class else self.act

        '''optimizer'''
        self.act_optimizer = torch.optim.Adam(self.act.parameters(), self.learning_rate)
        self.cri_optimizer = torch.optim.Adam(self.cri.parameters(), self.learning_rate) \
            if cri_class else self.act_optimizer

        """attribute"""
        if self.num_envs == 1:
            self.explore_env = self.explore_one_env
        else:
            self.explore_env = self.explore_vec_env

        self.if_use_per = getattr(args, 'if_use_per', None)  # use PER (Prioritized Experience Replay)
        self.criterion = torch.nn.SmoothL1Loss(reduction="mean")

        """save and load"""
        self.save_attr_names = {'act', 'act_target', 'act_optimizer', 'cri', 'cri_target', 'cri_optimizer'}

    def explore_one_env(self, env, horizon_len: int, if_random: bool = False) -> [Tensor]:
        """
        Collect trajectories through the actor-environment interaction for a **single** environment instance.

        env: RL training environment. env.reset() env.step(). It should be a vector env.
        horizon_len: collect horizon_len step while exploring to update networks
        if_random: uses random action for warn-up exploration
        return: `(states, actions, rewards, undones)` for off-policy
            num_envs == 1
            states.shape == (horizon_len, num_envs, state_dim)
            actions.shape == (horizon_len, num_envs, action_dim)
            rewards.shape == (horizon_len, num_envs)
            undones.shape == (horizon_len, num_envs)
        """
        states = torch.zeros((horizon_len, self.num_envs, self.state_dim), dtype=torch.float32).to(self.device)
        actions = torch.zeros((horizon_len, self.num_envs, 1), dtype=torch.int32).to(self.device) if self.if_discrete \
            else torch.zeros((horizon_len, self.num_envs, self.action_dim), dtype=torch.float32).to(self.device)
        rewards = torch.zeros((horizon_len, self.num_envs), dtype=torch.float32).to(self.device)
        dones = torch.zeros((horizon_len, self.num_envs), dtype=torch.bool).to(self.device)

        state = self.last_state  # last_state.shape = (state_dim, ) for a single env.
        get_action = self.act.get_action
        for t in range(horizon_len):
            action = torch.rand(1, self.action_dim) * 2 - 1.0 if if_random \
                else get_action(state.unsqueeze(0))
            states[t] = state

            ary_action = action[0].detach().cpu().numpy()
            ary_state, reward, done, _ = env.step(ary_action)  # next_state
            state = torch.as_tensor(env.reset() if done else ary_state,
                                    dtype=torch.float32, device=self.device)
            actions[t] = action
            rewards[t] = reward
            dones[t] = done

        self.last_state = state

        rewards *= self.reward_scale
        undones = 1.0 - dones.type(torch.float32)
        return states, actions, rewards, undones

    def explore_vec_env(self, env, horizon_len: int, if_random: bool = False) -> [Tensor]:
        """
        Collect trajectories through the actor-environment interaction for a **vectorized** environment instance.

        env: RL training environment. env.reset() env.step(). It should be a vector env.
        horizon_len: collect horizon_len step while exploring to update networks
        if_random: uses random action for warn-up exploration
        return: `(states, actions, rewards, undones)` for off-policy
            states.shape == (horizon_len, num_envs, state_dim)
            actions.shape == (horizon_len, num_envs, action_dim)
            rewards.shape == (horizon_len, num_envs)
            undones.shape == (horizon_len, num_envs)
        """
        states = torch.zeros((horizon_len, self.num_envs, self.state_dim), dtype=torch.float32).to(self.device)
        actions = torch.zeros((horizon_len, self.num_envs, 1), dtype=torch.int32).to(self.device) if self.if_discrete \
            else torch.zeros((horizon_len, self.num_envs, self.action_dim), dtype=torch.float32).to(self.device)
        rewards = torch.zeros((horizon_len, self.num_envs), dtype=torch.float32).to(self.device)
        dones = torch.zeros((horizon_len, self.num_envs), dtype=torch.bool).to(self.device)

        state = self.last_state  # last_state.shape = (num_envs, state_dim) for a vectorized env.

        get_action = self.act.get_action
        for t in range(horizon_len):
            action = torch.rand(self.num_envs, self.action_dim) * 2 - 1.0 if if_random \
                else get_action(state).detach()
            states[t] = state

            state, reward, done, _ = env.step(action)  # next_state
            actions[t] = action
            rewards[t] = reward
            dones[t] = done

        self.last_state = state

        rewards *= self.reward_scale
        undones = 1.0 - dones.type(torch.float32)
        return states, actions, rewards, undones

    def update_avg_std_for_state_value_norm(self, states: Tensor, returns: Tensor):
        tau = self.state_value_tau

        state_avg = states.mean(dim=0, keepdim=True)
        state_vam = (states ** 2).mean(dim=0, keepdim=True)
        self.cri.state_avg[:] = self.cri.state_avg * (1 - tau) + state_avg * tau
        self.cri.state_vam[:] = self.cri.state_vam * (1 - tau) + state_vam * tau
        self.cri.state_std[:] = torch.sqrt(self.cri.state_vam - self.cri.state_avg ** 2) + 1e-4

        returns_avg = returns.mean(dim=0)
        returns_vam = (returns ** 2).mean(dim=0)
        self.cri.value_avg[:] = self.cri.value_avg * (1 - tau) + returns_avg * tau
        self.cri.value_vam[:] = self.cri.value_vam * (1 - tau) + returns_vam * tau
        self.cri.value_std[:] = torch.sqrt(self.cri.value_vam - self.cri.value_avg ** 2) + 1e-4

        self.act.state_avg[:] = self.cri.state_avg
        self.act.state_std[:] = self.cri.state_std

    def optimizer_update(self, optimizer: torch.optim, objective: Tensor):
        """minimize the optimization objective via update the network parameters

        optimizer: `optimizer = torch.optim.SGD(net.parameters(), learning_rate)`
        objective: `objective = net(...)` the optimization objective, sometimes is a loss function.
        """
        optimizer.zero_grad()
        objective.backward()
        clip_grad_norm_(parameters=optimizer.param_groups[0]["params"], max_norm=self.clip_grad_norm)
        optimizer.step()

    def optimizer_update_amp(self, optimizer, objective: Tensor):  # automatic mixed precision
        """minimize the optimization objective via update the network parameters

        amp: Automatic Mixed Precision

        optimizer: `optimizer = torch.optim.SGD(net.parameters(), learning_rate)`
        objective: `objective = net(...)` the optimization objective, sometimes is a loss function.
        """
        amp_scale = torch.cuda.amp.GradScaler()  # write in __init__()

        optimizer.zero_grad()
        amp_scale.scale(objective).backward()  # loss.backward()
        amp_scale.unscale_(optimizer)  # amp

        # from torch.nn.utils import clip_grad_norm_
        clip_grad_norm_(parameters=optimizer.param_groups[0]["params"], max_norm=self.clip_grad_norm)
        amp_scale.step(optimizer)  # optimizer.step()
        amp_scale.update()  # optimizer.step()

    @staticmethod
    def soft_update(target_net: torch.nn.Module, current_net: torch.nn.Module, tau: float):
        """soft update target network via current network

        target_net: update target network via current network to make training more stable.
        current_net: current network update via an optimizer
        tau: tau of soft target update: `target_net = target_net * (1-tau) + current_net * tau`
        """
        for tar, cur in zip(target_net.parameters(), current_net.parameters()):
            tar.data.copy_(cur.data * tau + tar.data * (1.0 - tau))

    def save_or_load_agent(self, cwd: str, if_save: bool):
        """save or load training files for Agent

        cwd: Current Working Directory. ElegantRL save training files in CWD.
        if_save: True: save files. False: load files.
        """
        assert self.save_attr_names.issuperset({'act', 'act_target', 'act_optimizer'})

        for attr_name in self.save_attr_names:
            save_path = f"{cwd}/{attr_name}.pth"
            if if_save:
                torch.save(getattr(self, attr_name), save_path)
            elif os.path.isfile(save_path):
                setattr(self, attr_name, torch.load(save_path, map_location=self.device))


class AgentPPO(AgentBase):
    """
    PPO algorithm. “Proximal Policy Optimization Algorithms”. John Schulman. et al.. 2017.

    net_dims: the middle layer dimension of MLP (MultiLayer Perceptron)
    state_dim: the dimension of state (the number of state vector)
    action_dim: the dimension of action (or the number of discrete action)
    gpu_id: the gpu_id of the training device. Use CPU when cuda is not available.
    args: the arguments for agent training. `args = Config()`
    """

    def __init__(self, net_dims: [int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        self.if_off_policy = False
        self.act_class = getattr(self, "act_class", ActorPPO)
        self.cri_class = getattr(self, "cri_class", CriticPPO)
        AgentBase.__init__(self, net_dims, state_dim, action_dim, gpu_id, args)

        self.ratio_clip = getattr(args, "ratio_clip", 0.25)  # `ratio.clamp(1 - clip, 1 + clip)`
        self.lambda_gae_adv = getattr(args, "lambda_gae_adv", 0.95)  # could be 0.50~0.99 # GAE for sparse reward
        self.lambda_entropy = getattr(args, "lambda_entropy", 0.01)  # could be 0.00~0.20
        self.lambda_entropy = torch.tensor(self.lambda_entropy, dtype=torch.float32, device=self.device)

        if getattr(args, 'if_use_v_trace', False):
            self.get_advantages = self.get_advantages_vtrace  # get advantage value in reverse time series (V-trace)
        else:
            self.get_advantages = self.get_advantages_origin  # get advantage value using critic network
        self.value_avg = torch.zeros(1, dtype=torch.float32, device=self.device)
        self.value_std = torch.ones(1, dtype=torch.float32, device=self.device)

    def explore_one_env(self, env, horizon_len: int, if_random: bool = False) -> [Tensor]:
        """
        Collect trajectories through the actor-environment interaction for a **single** environment instance.

        env: RL training environment. env.reset() env.step(). It should be a vector env.
        horizon_len: collect horizon_len step while exploring to update networks
        return: `(states, actions, rewards, undones)` for off-policy
            env_num == 1
            states.shape == (horizon_len, env_num, state_dim)
            actions.shape == (horizon_len, env_num, action_dim)
            logprobs.shape == (horizon_len, env_num, action_dim)
            rewards.shape == (horizon_len, env_num)
            undones.shape == (horizon_len, env_num)
        """
        states = torch.zeros((horizon_len, self.num_envs, self.state_dim), dtype=torch.float32).to(self.device)
        actions = torch.zeros((horizon_len, self.num_envs, 1), dtype=torch.int32).to(self.device) if self.if_discrete \
            else torch.zeros((horizon_len, self.num_envs, self.action_dim), dtype=torch.float32).to(self.device)
        logprobs = torch.zeros((horizon_len, self.num_envs), dtype=torch.float32).to(self.device)
        rewards = torch.zeros((horizon_len, self.num_envs), dtype=torch.float32).to(self.device)
        dones = torch.zeros((horizon_len, self.num_envs), dtype=torch.bool).to(self.device)

        state = self.last_state  # shape == (1, state_dim) for a single env.

        get_action = self.act.get_action
        convert = self.act.convert_action_for_env
        for t in range(horizon_len):
            action, logprob = get_action(state)
            states[t] = state

            ary_action = convert(action[0]).detach().cpu().numpy()
            ary_state, reward, done, _ = env.step(ary_action)  # next_state
            state = torch.as_tensor(env.reset() if done else ary_state,
                                    dtype=torch.float32, device=self.device).unsqueeze(0)
            actions[t] = action
            logprobs[t] = logprob
            rewards[t] = reward
            dones[t] = done

        self.last_state = state

        rewards *= self.reward_scale
        undones = 1.0 - dones.type(torch.float32)
        return states, actions, logprobs, rewards, undones

    def explore_vec_env(self, env, horizon_len: int, if_random: bool = False) -> [Tensor]:
        """
        Collect trajectories through the actor-environment interaction for a **vectorized** environment instance.

        env: RL training environment. env.reset() env.step(). It should be a vector env.
        horizon_len: collect horizon_len step while exploring to update networks
        return: `(states, actions, rewards, undones)` for off-policy
            states.shape == (horizon_len, env_num, state_dim)
            actions.shape == (horizon_len, env_num, action_dim)
            logprobs.shape == (horizon_len, env_num, action_dim)
            rewards.shape == (horizon_len, env_num)
            undones.shape == (horizon_len, env_num)
        """
        states = torch.zeros((horizon_len, self.num_envs, self.state_dim), dtype=torch.float32).to(self.device)
        actions = torch.zeros((horizon_len, self.num_envs, 1), dtype=torch.int32).to(self.device) if self.if_discrete \
            else torch.zeros((horizon_len, self.num_envs, self.action_dim), dtype=torch.float32).to(self.device)
        logprobs = torch.zeros((horizon_len, self.num_envs), dtype=torch.float32).to(self.device)
        rewards = torch.zeros((horizon_len, self.num_envs), dtype=torch.float32).to(self.device)
        dones = torch.zeros((horizon_len, self.num_envs), dtype=torch.bool).to(self.device)

        state = self.last_state  # shape == (env_num, state_dim) for a vectorized env.

        get_action = self.act.get_action
        convert = self.act.convert_action_for_env
        for t in range(horizon_len):
            action, logprob = get_action(state)
            states[t] = state

            state, reward, done, _ = env.step(convert(action))  # next_state
            actions[t] = action
            logprobs[t] = logprob
            rewards[t] = reward
            dones[t] = done

        self.last_state = state

        rewards *= self.reward_scale
        undones = 1.0 - dones.type(torch.float32)
        return states, actions, logprobs, rewards, undones

    def update_net(self, buffer) -> [float]:
        with torch.no_grad():
            states, actions, logprobs, rewards, undones = buffer
            buffer_size = states.shape[0]
            buffer_num = states.shape[1]

            '''get advantages and reward_sums'''
            bs = 2 ** 10  # set a smaller 'batch_size' to avoiding out of GPU memory.
            values = torch.empty_like(rewards)  # values.shape == (buffer_size, buffer_num)
            for i in range(0, buffer_size, bs):
                for j in range(buffer_num):
                    values[i:i + bs, j] = self.cri(states[i:i + bs, j]).squeeze(1)

            advantages = self.get_advantages(rewards, undones, values)  # shape == (buffer_size, buffer_num)
            reward_sums = advantages + values  # shape == (buffer_size, buffer_num)
            del rewards, undones, values

            advantages = (advantages - advantages.mean()) / (advantages.std(dim=0) + 1e-4)

            self.update_avg_std_for_state_value_norm(
                states=states.reshape((-1, self.state_dim)),
                returns=reward_sums.reshape((-1,))
            )  # todo
        # assert logprobs.shape == advantages.shape == reward_sums.shape == (buffer_size, buffer_num)

        '''update network'''
        obj_critics = 0.0
        obj_actors = 0.0
        sample_len = buffer_size - 1

        update_times = int(sample_len * buffer_num * self.repeat_times / self.batch_size)
        assert update_times >= 1
        for _ in range(update_times):
            ids = torch.randint(sample_len * buffer_num, size=(self.batch_size,), requires_grad=False)
            ids0 = torch.fmod(ids, sample_len)  # ids % sample_len
            ids1 = torch.div(ids, sample_len, rounding_mode='floor')  # ids // sample_len

            state = states[ids0, ids1]
            action = actions[ids0, ids1]
            logprob = logprobs[ids0, ids1]
            advantage = advantages[ids0, ids1]
            reward_sum = reward_sums[ids0, ids1]

            value = self.cri(state).squeeze(1)  # critic network predicts the reward_sum (Q value) of state
            obj_critic = self.criterion(value, reward_sum)
            self.optimizer_update(self.cri_optimizer, obj_critic)

            new_logprob, obj_entropy = self.act.get_logprob_entropy(state, action)
            ratio = (new_logprob - logprob.detach()).exp()
            surrogate1 = advantage * ratio
            surrogate2 = advantage * ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)
            obj_surrogate = torch.min(surrogate1, surrogate2).mean()

            obj_actor = obj_surrogate + obj_entropy.mean() * self.lambda_entropy
            self.optimizer_update(self.act_optimizer, -obj_actor)

            obj_critics += obj_critic.item()
            obj_actors += obj_actor.item()
        a_std_log = self.act.action_std_log.mean()
        return obj_critics / update_times, obj_actors / update_times, a_std_log.item()

    def get_advantages_origin(self, rewards: Tensor, undones: Tensor, values: Tensor) -> Tensor:
        advantages = torch.empty_like(values)  # advantage value

        masks = undones * self.gamma
        horizon_len = rewards.shape[0]

        next_value = self.cri(self.last_state).detach().squeeze(1)

        advantage = torch.zeros_like(next_value)  # last advantage value by GAE (Generalized Advantage Estimate)
        for t in range(horizon_len - 1, -1, -1):
            next_value = rewards[t] + masks[t] * next_value
            advantages[t] = advantage = next_value - values[t] + masks[t] * self.lambda_gae_adv * advantage
            next_value = values[t]
        return advantages

    def get_advantages_vtrace(self, rewards: Tensor, undones: Tensor, values: Tensor) -> Tensor:
        advantages = torch.empty_like(values)  # advantage value

        masks = undones * self.gamma
        horizon_len = rewards.shape[0]

        advantage = torch.zeros_like(values[0])  # last advantage value by GAE (Generalized Advantage Estimate)
        for t in range(horizon_len - 1, -1, -1):
            advantages[t] = rewards[t] - values[t] + masks[t] * advantage
            advantage = values[t] + self.lambda_gae_adv * advantages[t]
        return advantages
