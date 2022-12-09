import os
import time
import multiprocessing as mp
import torch
import numpy as np
from copy import deepcopy

from elegantrl.agent import AgentBase
from elegantrl.config import Config, build_env
from elegantrl.evaluator import Evaluator

if os.name == 'nt':  # if is WindowOS
    """Fix bug about Anaconda in WindowOS
    OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
    """
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def train_agent(args: Config):
    args.init_before_training()
    torch.set_grad_enabled(False)

    '''init environment'''
    env = build_env(args.env_class, args.env_args, args.gpu_id)

    '''init agent'''
    agent = args.agent_class(args.net_dims, args.state_dim, args.action_dim, gpu_id=args.gpu_id, args=args)
    agent.save_or_load_agent(args.cwd, if_save=False)

    '''init agent.last_state'''
    state = env.reset()
    if args.num_envs == 1:
        assert state.shape == (args.state_dim,)
        assert isinstance(state, np.ndarray)
        state = torch.tensor(env.reset(), dtype=torch.float32, device=agent.device).unsqueeze(0)
    else:
        assert state.shape == (args.num_envs, args.state_dim)
        assert isinstance(state, torch.Tensor)
        state = env.reset().to(agent.device)
    assert state.shape == (args.num_envs, args.state_dim)
    assert isinstance(state, torch.Tensor)
    agent.last_state = state.detach()

    '''init buffer'''
    buffer = []

    '''init evaluator'''
    eval_env_class = args.eval_env_class if args.eval_env_class else args.env_class
    eval_env_args = args.eval_env_args if args.eval_env_args else args.env_args
    eval_env = build_env(eval_env_class, eval_env_args, args.gpu_id)
    evaluator = Evaluator(cwd=args.cwd, agent_id=0, eval_env=eval_env, args=args, if_tensorboard=False)

    '''train loop'''
    cwd = args.cwd
    break_step = args.break_step
    horizon_len = args.horizon_len
    del args

    if_train = True
    while if_train:
        buffer_items = agent.explore_env(env, horizon_len)
        exp_r = buffer_items[2].mean().item()
        buffer[:] = buffer_items

        torch.set_grad_enabled(True)
        logging_tuple = agent.update_net(buffer)
        torch.set_grad_enabled(False)

        evaluator.evaluate_and_save(actor=agent.act, steps=horizon_len, exp_r=exp_r, logging_tuple=logging_tuple)
        if_train = (evaluator.total_step <= break_step) and (not os.path.exists(f"{cwd}/stop"))

    print(f'| UsedTime: {time.time() - evaluator.start_time:>7.0f} | SavedDir: {cwd}')
    evaluator.save_training_curve_jpg()
    agent.save_or_load_agent(cwd, if_save=True)


'''train multiple process'''


def train_agent_multiprocessing(args: Config):
    args.init_before_training()

    mp.set_start_method(method='spawn', force=True)  # force all the multiprocessing to 'spawn' methods

    evaluator_pipe = PipeEvaluator()
    evaluator_proc = mp.Process(target=evaluator_pipe.run, args=(args,))

    worker_pipe = PipeWorker(args.num_workers)
    worker_procs = [mp.Process(target=worker_pipe.run, args=(args, worker_id))
                    for worker_id in range(args.num_workers)]

    learner_pipe = PipeLearner()
    learner_proc = mp.Process(target=learner_pipe.run, args=(args, evaluator_pipe, worker_pipe))

    process = worker_procs + [learner_proc, evaluator_proc]
    [p.start() for p in process]
    [p.join() for p in process]


class PipeWorker:
    def __init__(self, worker_num: int):
        self.worker_num = worker_num
        pipes = [mp.Pipe() for _ in range(worker_num)]
        self.pipe0s = [pipe[0] for pipe in pipes]
        self.pipe1s = [pipe[1] for pipe in pipes]

    def explore(self, agent: AgentBase):
        actor = agent.act

        for worker_id in range(self.worker_num):
            self.pipe1s[worker_id].send(actor)

        recv_items = [pipe1.recv() for pipe1 in self.pipe1s]
        buffer_items, last_state = [item for item in zip(*recv_items)]

        buffer_items = [torch.cat(tensors, dim=1) for tensors in zip(*buffer_items)]
        """buffer_items
        states, actions, rewards, dones = buffer_items  # off-policy
        states, actions, logprobs, rewards, dones = buffer_items  # on-policy

        assert states.shape == (horizon_len, num_envs * worker_num, state_dim)
        assert actions.shape == (horizon_len, num_envs * worker_num, action_dim)
        assert logprobs.shape == (horizon_len, num_envs * worker_num, action_dim)
        assert rewards.shape == (horizon_len, num_envs * worker_num)
        assert dones.shape == (horizon_len, num_envs * worker_num)  
        """

        last_state = torch.cat(last_state, dim=0)
        """last_state
        assert last_state.shape == (num_envs * worker_num, state_dim)
        """
        return buffer_items, last_state

    def run(self, args: Config, worker_id: int):
        torch.set_grad_enabled(False)

        '''init environment'''
        env = build_env(args.env_class, args.env_args, args.gpu_id)

        '''init agent'''
        agent = args.agent_class(args.net_dims, args.state_dim, args.action_dim, gpu_id=args.gpu_id, args=args)
        agent.save_or_load_agent(args.cwd, if_save=False)

        '''init agent.last_state'''
        state = env.reset()
        if args.num_envs == 1:
            assert state.shape == (args.state_dim,)
            assert isinstance(state, np.ndarray)
            state = torch.tensor(env.reset(), dtype=torch.float32, device=agent.device).unsqueeze(0)
        else:
            assert state.shape == (args.num_envs, args.state_dim)
            assert isinstance(state, torch.Tensor)
            state = env.reset().to(agent.device)
        assert state.shape == (args.num_envs, args.state_dim)
        assert isinstance(state, torch.Tensor)
        agent.last_state = state.detach()

        '''init buffer'''
        horizon_len = args.horizon_len
        if args.if_off_policy:
            buffer_items = agent.explore_env(env, args.horizon_len * args.eval_times, if_random=True)
            self.pipe0s[worker_id].send((buffer_items, agent.last_state))
        del args

        '''loop'''
        while True:
            actor = self.pipe0s[worker_id].recv()
            if actor is None:
                break

            agent.act = actor
            buffer_items = agent.explore_env(env, horizon_len)
            self.pipe0s[worker_id].send((buffer_items, agent.last_state))

        '''close pipe1s'''
        while self.pipe1s[worker_id].poll():
            time.sleep(1)
            self.pipe1s[worker_id].recv()

        if hasattr(env, 'close'):
            env.close()


class PipeEvaluator:
    def __init__(self):
        self.pipe0, self.pipe1 = mp.Pipe()

    def evaluate_and_save(self, actor, steps: int, r_exp: float, logging_tuple: tuple) -> (bool, bool):
        if self.pipe1.poll():  # if_evaluator_idle
            if_train = self.pipe1.recv()
            actor = deepcopy(actor)
        else:
            if_train = True
            actor = None

        self.pipe1.send((actor, steps, r_exp, logging_tuple))
        return if_train

    def run(self, args: Config):
        torch.set_grad_enabled(False)
        # import wandb
        # wandb_project_name = "RL_training"
        # wandb.init(project=wandb_project_name)

        '''init evaluator'''
        eval_env_class = args.eval_env_class if args.eval_env_class else args.env_class
        eval_env_args = args.eval_env_args if args.eval_env_args else args.env_args
        eval_env = build_env(eval_env_class, eval_env_args, args.gpu_id)
        evaluator = Evaluator(cwd=args.cwd, agent_id=args.gpu_id, eval_env=eval_env, args=args, if_tensorboard=False)

        '''loop'''
        cwd = args.cwd
        break_step = args.break_step
        del args

        if_train = True
        while if_train:
            pipe0_recv = self.pipe0.recv()
            actor, steps, exp_r, logging_tuple = pipe0_recv
            # wandb.log({"obj_cri": logging_tuple[0], "obj_act": logging_tuple[1]})

            if actor is None:
                evaluator.total_step += steps  # update total_step but don't update recorder
                if_train = True
            else:
                evaluator.evaluate_and_save(actor, steps, exp_r, logging_tuple)
                if_train = (evaluator.total_step <= break_step) and (not os.path.exists(f"{cwd}/stop"))
            self.pipe0.send(if_train)

        evaluator.save_training_curve_jpg()
        print(f'| TrainingTime: {time.time() - evaluator.start_time:>7.0f} | SavedDir: {cwd}')

        if hasattr(eval_env, 'close'):
            eval_env.close()

        '''close pipe1'''
        while self.pipe1.poll():
            time.sleep(1)


class PipeLearner:
    def __init__(self):
        pass

    @staticmethod
    def run(args: Config, comm_eva: PipeEvaluator, comm_exp: PipeWorker):
        torch.set_grad_enabled(False)

        '''init agent'''
        agent = args.agent_class(args.net_dims, args.state_dim, args.action_dim, gpu_id=args.gpu_id, args=args)
        agent.save_or_load_agent(args.cwd, if_save=False)

        '''init buffer'''
        buffer = []

        '''loop'''
        if_save_buffer = args.if_save_buffer
        steps = args.horizon_len * args.num_workers
        cwd = args.cwd
        del args

        if_train = True
        while if_train:
            buffer_items, last_state = comm_exp.explore(agent)
            exp_r = buffer_items[2].mean().item()
            buffer[:] = buffer_items
            agent.last_state = last_state

            torch.set_grad_enabled(True)
            logging_tuple = agent.update_net(buffer)
            torch.set_grad_enabled(False)

            if_train = comm_eva.evaluate_and_save(agent.act, steps, exp_r, logging_tuple)

        agent.save_or_load_agent(cwd, if_save=True)
        # print(f'| Learner: Save in {cwd}')

        if if_save_buffer and hasattr(buffer, 'save_or_load_history'):
            print(f"| LearnerPipe.run: ReplayBuffer saving in {cwd}")
            buffer.save_or_load_history(cwd, if_save=True)
            print(f"| LearnerPipe.run: ReplayBuffer saved  in {cwd}")

        '''comm_exp: close pipe1'''
        for j, pipe1 in enumerate(comm_exp.pipe1s):
            while pipe1.poll():
                time.sleep(1)
                pipe1.recv()
        for j, pipe1 in enumerate(comm_exp.pipe1s):
            pipe1.send(None)

        '''comm_exp: waiting for closing pipe0'''
        for j, pipe0 in enumerate(comm_exp.pipe0s):
            while pipe0.poll():
                time.sleep(1)

        '''comm_eva: close pipe1'''
        while comm_eva.pipe1.poll():
            time.sleep(1)
            comm_eva.pipe1.recv()
