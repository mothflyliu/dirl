import logging
import numpy as np
from mjrl.utils.gym_env import GymEnv
from mjrl.utils import tensor_utils
logging.disable(logging.CRITICAL)
import multiprocessing as mp
import time as timer
logging.disable(logging.CRITICAL)


# Single core rollout to sample trajectories
# =======================================================
def do_rollout(
        num_traj,
        env,
        policy,
        eval_mode = False,
        horizon = 1e6,
        base_seed = None,
        env_kwargs=None,
):
    """
    :param num_traj:    number of trajectories (int)
    :param env:         environment (env class, str with env_name, or factory function)
    :param policy:      policy to use for action selection
    :param eval_mode:   use evaluation mode for action computation (bool)
    :param horizon:     max horizon length for rollout (<= env.horizon)
    :param base_seed:   base seed for rollouts (int)
    :param env_kwargs:  dictionary with parameters, will be passed to env generator
    :return:
    """

    # get the correct env behavior
    if type(env) == str:
        env = GymEnv(env)
    elif isinstance(env, GymEnv):
        env = env
    elif callable(env):
        env = env(**env_kwargs)
    else:
        print("Unsupported environment format")
        raise AttributeError

    if base_seed is not None:
        env.set_seed(base_seed)
        np.random.seed(base_seed)
    else:
        np.random.seed()
    horizon = min(horizon, env.horizon)
    paths = []

    for ep in range(num_traj):
        # seeding
        if base_seed is not None:
            seed = base_seed + ep
            env.set_seed(seed)
            np.random.seed(seed)

        observations=[]
        actions=[]
        rewards=[]
        agent_infos = []
        env_infos = []

        o = env.reset()
        done = False
        t = 0

        while t < horizon and done != True:
            a, agent_info = policy.get_action(o)
            if eval_mode:
                a = agent_info['evaluation']
            env_info_base = env.get_env_infos()
            next_o, r, done, env_info_step = env.step(a)
            # below is important to ensure correct env_infos for the timestep
            env_info = env_info_step if env_info_base == {} else env_info_base
            observations.append(o)
            actions.append(a)
            rewards.append(r)
            agent_infos.append(agent_info)
            env_infos.append(env_info)
            o = next_o
            t += 1

        path = dict(
            observations=np.array(observations),
            actions=np.array(actions),
            rewards=np.array(rewards),
            agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos),
            env_infos=tensor_utils.stack_tensor_dict_list(env_infos),
            terminated=done
        )
        paths.append(path)

    del(env)
    return paths


def sample_paths(
        num_traj,
        env,
        policy,
        eval_mode = False,
        horizon = 1e6,
        base_seed = None,
        num_cpu = 1,
        max_process_time=300,
        max_timeouts=4,
        suppress_print=False,
        env_kwargs=None,
        ):

    num_cpu = 1 if num_cpu is None else num_cpu
    num_cpu = mp.cpu_count() if num_cpu == 'max' else num_cpu
    assert type(num_cpu) == int

    if num_cpu == 1:
        input_dict = dict(num_traj=num_traj, env=env, policy=policy,
                          eval_mode=eval_mode, horizon=horizon, base_seed=base_seed,
                          env_kwargs=env_kwargs)
        # dont invoke multiprocessing if not necessary
        return do_rollout(**input_dict)

    # do multiprocessing otherwise
    paths_per_cpu = int(np.ceil(num_traj/num_cpu))
    input_dict_list= []
    for i in range(num_cpu):
        input_dict = dict(num_traj=paths_per_cpu, env=env, policy=policy,
                          eval_mode=eval_mode, horizon=horizon,
                          base_seed=base_seed + i * paths_per_cpu,
                          env_kwargs=env_kwargs)
        input_dict_list.append(input_dict)
    if suppress_print is False:
        start_time = timer.time()
        print("####### Gathering Samples #######")

    results = _try_multiprocess(do_rollout, input_dict_list,
                                num_cpu, max_process_time, max_timeouts)
    paths = []
    # result is a paths type and results is list of paths
    for result in results:
        for path in result:
            paths.append(path)  

    if suppress_print is False:
        print("======= Samples Gathered  ======= | >>>> Time taken = %f " %(timer.time()-start_time) )

    return paths


def sample_paths(
        num_traj,
        env,
        policy,
        eval_mode = False,
        horizon = 1e6,
        base_seed = None,
        num_cpu = 1,
        max_process_time=300,
        max_timeouts=4,
        suppress_print=False,
        env_kwargs=None,
        ):
    # 如果 num_cpu 未设置或为 None，设置为 1
    num_cpu = 1 if num_cpu is None else num_cpu
    # 如果 num_cpu 为'max'，设置为 CPU 核心数
    num_cpu = mp.cpu_count() if num_cpu == 'ax' else num_cpu
    # 确保 num_cpu 是整数类型
    assert type(num_cpu) == int

    # 如果 num_cpu 为 1，直接执行单个进程的采样操作
    if num_cpu == 1:
        input_dict = dict(num_traj=num_traj, env=env, policy=policy,
                          eval_mode=eval_mode, horizon=horizon, base_seed=base_seed,
                          env_kwargs=env_kwargs)
        # 不启用多进程，如果不必要
        return do_rollout(**input_dict)

    # 否则，进行多进程处理
    paths_per_cpu = int(np.ceil(num_traj/num_cpu))
    input_dict_list= []
    for i in range(num_cpu):
        # 为每个 CPU 分配采样任务的参数
        input_dict = dict(num_traj=paths_per_cpu, env=env, policy=policy,
                          eval_mode=eval_mode, horizon=horizon,
                          base_seed=base_seed + i * paths_per_cpu,
                          env_kwargs=env_kwargs)
        input_dict_list.append(input_dict)
    # 如果不抑制打印
    if suppress_print is False:
        start_time = timer.time()
        print("####### Gathering Samples #######")

    # 尝试多进程执行采样操作jenny
    results = _try_multiprocess(do_rollout, input_dict_list,
                                num_cpu, max_process_time, max_timeouts)
    paths = []
    # 处理多进程的结果
    for result in results:
        for path in result:
            paths.append(path)

    # 如果不抑制打印
    if suppress_print is False:
        print("======= Samples Gathered  ======= | >>>> Time taken = %f " %(timer.time()-start_time) )

    return paths


def _try_multiprocess(func, input_dict_list, num_cpu, max_process_time, max_timeouts):
    # 基本情况：如果最大超时次数为 0，返回 None
    if max_timeouts == 0:
        return None

    # 创建一个进程池，指定进程数量
    pool = mp.Pool(processes=num_cpu, maxtasksperchild=None)
    # 为每个输入字典创建异步任务
    parallel_runs = [pool.apply_async(func, kwds=input_dict) for input_dict in input_dict_list]

    try:
        # 尝试在指定的超时时间内获取每个异步任务的结果
        results = [p.get(timeout=max_process_time) for p in parallel_runs]
    except Exception as e:
        # 如果出现超时错误
        print(str(e))
        print("Timeout Error raised... Trying again")
        # 关闭、终止并等待进程池结束
        pool.close()
        pool.terminate()
        pool.join()
        # 递归调用自身，减少超时次数并再次尝试
        return _try_multiprocess(func, input_dict_list, num_cpu, max_process_time, max_timeouts - 1)

    # 关闭、终止并等待进程池结束
    pool.close()
    pool.terminate()
    pool.join()
    # 返回获取到的结果
    return results
