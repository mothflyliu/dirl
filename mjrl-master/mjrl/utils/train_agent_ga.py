import logging
logging.disable(logging.CRITICAL)

from tabulate import tabulate
from mjrl.utils.make_train_plots import make_train_plots
from mjrl.utils.gym_env import GymEnv
from mjrl.samplers.core import sample_paths
import numpy as np
import pickle
import time as timer
import os
import copy


def _load_latest_policy_and_logs(agent, *, policy_dir, logs_dir):
    """Loads the latest policy.
    Returns the next step number to begin with.
    """
    assert os.path.isdir(policy_dir), str(policy_dir)
    assert os.path.isdir(logs_dir), str(logs_dir)

    log_csv_path = os.path.join(logs_dir, 'log.csv')
    if not os.path.exists(log_csv_path):
        return 0   # fresh start

    print("Reading: {}".format(log_csv_path))
    agent.logger.read_log(log_csv_path)
    last_step = agent.logger.max_len - 1
    if last_step <= 0:
        return 0   # fresh start


    # find latest policy/baseline
    i = last_step
    while i >= 0:
        policy_path = os.path.join(policy_dir, 'policy_{}.pickle'.format(i))
        baseline_path = os.path.join(policy_dir, 'baseline_{}.pickle'.format(i))

        if not os.path.isfile(policy_path):
            i = i -1
            continue
        else:
            print("Loaded last saved iteration: {}".format(i))

        with open(policy_path, 'rb') as fp:
            agent.policy = pickle.load(fp)
        with open(baseline_path, 'rb') as fp:
            agent.baseline = pickle.load(fp)

        # additional
        # global_status_path = os.path.join(policy_dir, 'global_status.pickle')
        # with open(global_status_path, 'rb') as fp:
        #     agent.load_global_status( pickle.load(fp) )

        agent.logger.shrink_to(i + 1)
        assert agent.logger.max_len == i + 1
        return agent.logger.max_len

    # cannot find any saved policy
    raise RuntimeError("Log file exists, but cannot find any saved policy.")

def train_agent(job_name, agent,
                p=None,
                e=None,
                seed=0,
                a=0,
                niter=101,
                gamma=0.995,
                gae_lambda=None,
                num_cpu=1,
                sample_mode='trajectories',
                num_traj=50,
                num_samples=50000, # has precedence, used with sample_mode = 'amples'
                save_freq=10,
                evaluation_rollouts=None,
                plot_keys=['stoc_pol_mean'],
                ):
    if a == 0:
        # 如果指定的工作目录不存在，创建它
        if os.path.isdir(job_name) == False:
            os.mkdir(job_name)
        # 保存当前工作目录
        previous_dir = os.getcwd()
        # 切换到指定的工作目录
        os.chdir(job_name)
        # 如果'iterations'子目录不存在，创建它
        if os.path.isdir('iterations') == False: os.mkdir('iterations')
        # 如果需要保存日志且'logs'子目录不存在，创建它
        if os.path.isdir('logs') == False and agent.save_logs == True: os.mkdir('logs')

        # 深度复制初始策略作为最佳策略
        best_policy = copy.deepcopy(agent.policy)
        # 初始化最佳性能为一个非常小的值
        best_perf = -1e8
        # 初始化训练曲线，初始值为最佳性能
        train_curve = best_perf * np.ones(100)
        # 平均策略性能初始化为 0
        mean_pol_perf = 0.0
        # 尝试从现有检查点加载策略、日志等
        i_start = _load_latest_policy_and_logs(agent,
                                               policy_dir='iterations',
                                               logs_dir='logs')
        # 如果能够加载，表示恢复已有工作
        if i_start:
            print("Resuming from an existing job folder...")
    if a != 0:
        i_start = 0
        # 保存当前工作目录
        previous_dir = os.getcwd()
        # 切换到指定的工作目录
        os.chdir(previous_dir)
        # 重新运行文件时读取数据
        with open('best_perf.pkl', 'rb') as f:
            best_perf = pickle.load(f)
        with open('train_curve.pkl', 'rb') as f:
            train_curve = pickle.load(f)
    # 训练循环
    for i in range(i_start, 1):
        print("......................................................................................")
        print("ITERATION : %a " % a)

        # 如果当前训练曲线值优于最佳性能，更新最佳策略和最佳性能
        if train_curve[a-1] > best_perf:
            best_policy = copy.deepcopy(agent.policy)
            best_perf = train_curve[a-1]
            with open('best_policy.pkl', 'wb') as f:
                pickle.dump(best_policy, f)
        # 根据采样模式确定采样数量
        N = num_traj if sample_mode == 'trajectories' or 'GAIL' else num_samples
        # 构建训练参数
        args = dict(N=N, p=p, sample_mode=sample_mode, gamma=gamma, gae_lambda=gae_lambda, num_cpu=num_cpu)
        # 执行训练步骤并获取统计信息
        stats = agent.train_step(**args)
        # 记录当前迭代的训练曲线值
        train_curve[a] = stats[0]
        # 保存数据
        with open('best_perf.pkl', 'wb') as f:
            pickle.dump(best_perf, f)
        with open('train_curve.pkl', 'wb') as f:
            pickle.dump(train_curve, f)
        # 如果需要进行评估并且评估次数大于 0
        if evaluation_rollouts is not None and evaluation_rollouts > 0:
            print("Performing evaluation rollouts .......")
            # 采样评估路径
            eval_paths = sample_paths(num_traj=evaluation_rollouts, policy=agent.policy, num_cpu=num_cpu,
                                      env=e.env_id, eval_mode=True, base_seed=seed)
            # 计算平均策略性能
            mean_pol_perf = np.mean([np.sum(path['rewards']) for path in eval_paths])
            # 如果需要保存日志
            if agent.save_logs:
                agent.logger.log_kv('eval_score', mean_pol_perf)
                try:
                    # 尝试计算评估成功指标
                    eval_success = e.env.env.evaluate_success(eval_paths)
                    agent.logger.log_kv('eval_success', eval_success)
                except:
                    pass

        # 如果达到保存频率并且迭代次数大于 0
        if a % save_freq == 0 and a > 0:
            if agent.save_logs:
                # 保存日志
                agent.logger.save_log('logs/')
                # 生成训练图表
                make_train_plots(log=agent.logger.log, keys=plot_keys, save_loc='logs/')
            # 保存策略和基线
            policy_file = 'policy_%i.pickle' % a
            baseline_file = 'baseline_%i.pickle' % a
            pickle.dump(agent.policy, open('iterations/' + policy_file, 'wb'))
            pickle.dump(agent.baseline, open('iterations/' + baseline_file, 'wb'))
            with open('best_policy.pkl', 'rb') as f:
                best_policy = pickle.load(f)
            pickle.dump(best_policy, open('iterations/best_policy.pickle', 'wb'))

        # 打印结果到控制台和文件
        if a == 0:
            result_file = open('results.txt', 'w')
            print("Iter | Stoc Pol | Mean Pol | Best (Stoc) \n")
            result_file.write("Iter | Sampling Pol | Evaluation Pol | Best (Sampled) \n")
            result_file.close()
        print("[ %s ] %4i %5.2f %5.2f %5.2f " % (timer.asctime(timer.localtime(timer.time())),
                                                 a, train_curve[a], mean_pol_perf, best_perf))
        result_file = open('results.txt', 'a')
        result_file.write("%4i %5.2f %5.2f %5.2f \n" % (a, train_curve[a], mean_pol_perf, best_perf))
        result_file.close()
        if agent.save_logs:
            print_data = sorted(filter(lambda v: np.asarray(v[1]).size == 1,
                                       agent.logger.get_current_log().items()))
            print(tabulate(print_data))
    if a == 100:
        with open('best_policy.pkl', 'rb') as f:
            best_policy = pickle.load(f)
        # 最终保存最佳策略
        pickle.dump(best_policy, open('iterations/best_policy.pickle', 'wb'))
        # 如果需要保存日志，进行保存和生成图表
        if agent.save_logs:
            agent.logger.save_log('logs/')
            make_train_plots(log=agent.logger.log, keys=plot_keys, save_loc='logs/')
        # 切换回之前的工作目录
        os.chdir(previous_dir)
