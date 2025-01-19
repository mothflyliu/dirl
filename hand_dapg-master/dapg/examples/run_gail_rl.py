"""
This is a job script for running NPG/DAPG on hand tasks and other gym envs.
Note that DAPG generalizes PG and BC init + PG finetuning.
With appropriate settings of parameters, we can recover the full family.
"""

from mjrl.utils.gym_env import GymEnv
from mjrl.policies.gaussian_mlp import MLP
from mjrl.policies.Discriminator import Discriminator
from mjrl.baselines.quadratic_baseline import QuadraticBaseline
from mjrl.baselines.mlp_baseline import MLPBaseline
from mjrl.algos.npg_cg import NPG
from mjrl.algos.dapg import DAPG
from mjrl.algos.GAIL import GAIL
from mjrl.utils.train_agent import train_agent
from mjrl.samplers.core import sample_paths
import os
import json
import mjrl.envs
import mj_envs
import time as timer
import pickle
import argparse
import find
# ===============================================================================
# Get command line arguments
# ===============================================================================
if __name__ == '__main__':
    # 直接从用户输入获取存储结果的位置和配置文件路径
    # output_location = input("请输入存储结果的位置：")
    # config_file_path = input("请输入配置文件的路径：")
    output_location = './gail'
    config_file_path = 'gail.txt'
    # 将存储结果的位置赋给 JOB_DIR
    JOB_DIR = output_location
    # 如果指定的输出目录不存在，创建该目录
    if not os.path.exists(JOB_DIR):
        os.mkdir(JOB_DIR)
    # 打开配置文件并读取内容
    with open(config_file_path, 'r') as f:
        job_data = eval(f.read())
    # 断言'algorithm'在配置数据的键中
    assert 'algorithm' in job_data.keys()
    # 断言配置数据中的'algorithm'值是['NPG', 'BCRL', 'DAPG', 'GAIL', 'GDNPG', 'BGDNPG']中的一个
    assert any([job_data['algorithm'] == a for a in ['NPG', 'BCRL', 'DAPG', 'GAIL', 'GDNPG', 'BGDNPG']])
    # 如果'lam_0'不在配置数据的键中，将其设为 0.0，否则保持原值
    job_data['lam_0'] = 0.0 if 'lam_0' not in job_data.keys() else job_data['lam_0']
    # 同上，处理'lam_1'
    job_data['lam_1'] = 0.0 if 'lam_1' not in job_data.keys() else job_data['lam_1']
    # 定义实验文件路径
    EXP_FILE = JOB_DIR + '/job_config.json'
    # 将配置数据以 JSON 格式写入实验文件
    with open(EXP_FILE, 'w') as f:
        json.dump(job_data, f, indent=4)

    # ===============================================================================
    # Train Loop
    # ===============================================================================

    # 创建 Gym 环境
    e = GymEnv(job_data['env'])
    # 创建多层感知机策略
    policy = MLP(e.spec, hidden_sizes=job_data['policy_size'], seed=job_data['seed'])
    D = Discriminator(e.spec, hidden_sizes=job_data['policy_size'], seed=job_data['seed'])
    # 创建多层感知机基线
    baseline = MLPBaseline(e.spec, reg_coef=1e-3, batch_size=job_data['vf_batch_size'],
                           epochs=job_data['vf_epochs'], learn_rate=job_data['vf_learn_rate'])

    print("========================================")
    print("Collecting expert demonstrations")
    print("========================================")

    demo_file_name = job_data.get('demo_file', None)

    demo_paths = pickle.load(open(find.found_file_path(demo_file_name), 'rb'))
    # 创建强化学习代理

    agent = NPG(e, policy, baseline,
                    normalized_step_size=job_data['rl_step_size'],
                    seed=job_data['seed'], save_logs=True
                    )
    # 创建行为克隆代理
    GAIL_agent = GAIL(demo_paths, policy=policy, discriminator=D, epochs=job_data['epochs'], seed=job_data['seed'],
                      batch_size=job_data['batch_size'], lr=job_data['learn_rate'], loss_type='MSE',
                      set_transforms=False, job_name=JOB_DIR, niter=job_data['rl_num_iter'],
                      gamma=job_data['rl_gamma'], agent=agent, gae_lambda=job_data['rl_gae'],
                      num_cpu=job_data['num_cpu'], sample_mode='Discriminator', num_traj=job_data['rl_num_traj'])

    # 计算转换参数
    in_shift, in_scale, out_shift, out_scale = GAIL_agent.compute_transformations()
    # 设置转换参数
    GAIL_agent.set_transformations(in_shift, in_scale, out_shift, out_scale)
    # 根据数据设置方差
    GAIL_agent.set_variance_with_data(out_scale)

    ts = timer.time()
    print("========================================")
    print("Running GAIL with expert demonstrations")
    print("========================================")
    # 训练行为克隆代理
    GAIL_agent.train()
    print("========================================")
    print("GAIL training complete!!!")
    print("time taken = %f" % (timer.time() - ts))
    print("========================================")

    # 如果需要评估，计算策略得分
    if job_data['eval_rollouts'] >= 1:
        score = e.evaluate_policy(policy, num_episodes=job_data['eval_rollouts'], mean_action=True)
        print("Score with GAIL = %f" % score[0][0])

    # ===============================================================================
    # RL Loop
    # ===============================================================================
    # 创建强化学习代理

    rl_agent = NPG(e, policy, baseline,
                    normalized_step_size=job_data['rl_step_size'],
                    seed=job_data['seed'], save_logs=True
                    )
    print("========================================")
    print("Starting reinforcement learning phase")
    print("========================================")

    ts = timer.time()
    # 训练强化学习代理
    train_agent(job_name=JOB_DIR,
                agent=rl_agent,
                seed=job_data['seed'],
                niter=job_data['rl_num_iter'],
                gamma=job_data['rl_gamma'],
                gae_lambda=job_data['rl_gae'],
                num_cpu=job_data['num_cpu'],
                sample_mode='trajectories',
                num_traj=job_data['rl_num_traj'],
                save_freq=job_data['save_freq'],
                evaluation_rollouts=job_data['eval_rollouts'])
    print("time taken = %f" % (timer.time()-ts))