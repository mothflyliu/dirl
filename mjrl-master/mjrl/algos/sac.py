import logging
logging.disable(logging.CRITICAL)
import numpy as np
import scipy as sp
import scipy.sparse.linalg as spLA
import copy
import time as timer
import torch
import torch.nn as nn
from torch.autograd import Variable

# samplers
import mjrl.samplers.core as trajectory_sampler

# utility functions
import mjrl.utils.process_samples as process_samples
from mjrl.utils.logger import DataLog
from mjrl.utils.cg_solve import cg_solve


# Import Algs
from mjrl.algos.npg_cg import NPG
from mjrl.algos.behavior_cloning import BC


class SAC(NPG):
    def __init__(self, env, policy, baseline,
                 demo_paths=None,
                 normalized_step_size=0.01,
                 FIM_invert_args={'iters': 10, 'damping': 1e-4},
                 hvp_sample_frac=1.0,
                 seed=123,
                 save_logs=False,
                 kl_dist=None,
                 lam_0=1.0,  # demo coef
                 lam_1=0.95,  # decay coef
                 temperature=None,
                 value_lr=None,
                 sac_updates=None,
                 **kwargs):

        super().__init__(env, policy, baseline,
                         normalized_step_size=normalized_step_size,
                         FIM_invert_args=FIM_invert_args,
                         hvp_sample_frac=hvp_sample_frac,
                         seed=seed,
                         save_logs=save_logs,
                         kl_dist=kl_dist)

        self.demo_paths = demo_paths
        self.lam_0 = lam_0
        self.lam_1 = lam_1
        self.iter_count = 0.0
        if save_logs: self.logger = DataLog()

        # SAC-specific parameters
        self.sac_temperature = temperature if temperature is not None else 1.0
        self.value_function_lr = value_lr if value_lr is not None else 0.001
        self.sac_updates = sac_updates if sac_updates is not None else 10

        # SAC value functions
        self.state_value_function = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.action_value_function = nn.Sequential(
            nn.Linear(env.observation_space.shape[0] + env.action_space.shape[0], 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.target_state_value_function = copy.deepcopy(self.state_value_function)
        self.target_action_value_function = copy.deepcopy(self.action_value_function)
        self.value_function_optimizer = torch.optim.Adam(
            list(self.state_value_function.parameters()) + list(self.action_value_function.parameters()),
            lr=self.value_function_lr)

    def train_from_paths(self, paths):
        # Concatenate from all the trajectories
        observations = np.concatenate([path["observations"] for path in paths])
        actions = np.concatenate([path["actions"] for path in paths])
        advantages = np.concatenate([path["advantages"] for path in paths])
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-6)

        if self.demo_paths is not None and self.lam_0 > 0.0:
            demo_obs = np.concatenate([path["observations"] for path in self.demo_paths])
            demo_act = np.concatenate([path["actions"] for path in self.demo_paths])
            demo_adv = self.lam_0 * (self.lam_1 ** self.iter_count) * np.ones(demo_obs.shape[0])
            self.iter_count += 1
            all_obs = np.concatenate([observations, demo_obs])
            all_act = np.concatenate([actions, demo_act])
            all_adv = 1e-2 * np.concatenate([advantages / (np.std(advantages) + 1e-8), demo_adv])
        else:
            all_obs = observations
            all_act = actions
            all_adv = advantages

        # cache return distributions for the paths
        path_returns = [sum(p["rewards"]) for p in paths]
        mean_return = np.mean(path_returns)
        std_return = np.std(path_returns)
        min_return = np.amin(path_returns)
        max_return = np.amax(path_returns)
        base_stats = [mean_return, std_return, min_return, max_return]
        self.running_score = mean_return if self.running_score is None else \
            0.9 * self.running_score + 0.1 * mean_return

        if self.save_logs: self.log_rollout_statistics(paths)

        # Keep track of times for various computations
        t_gLL = 0.0
        t_FIM = 0.0
        sac_time = 0.0  # Initialize time for SAC updates

        # Optimization algorithm
        surr_before = self.CPI_surrogate(observations, actions, advantages).data.numpy().ravel()[0]

        # DAPG
        ts = timer.time()
        sample_coef = all_adv.shape[0] / advantages.shape[0]
        dapg_grad = sample_coef * self.flat_vpg(all_obs, all_act, all_adv)
        t_gLL += timer.time() - ts

        # NPG
        ts = timer.time()
        hvp = self.build_Hvp_eval([observations, actions],
                                  regu_coef=self.FIM_invert_args['damping'])
        npg_grad = cg_solve(hvp, dapg_grad, x_0=dapg_grad.copy(),
                            cg_iters=self.FIM_invert_args['iters'])
        t_FIM += timer.time() - ts

        # SAC updates
        ts = timer.time()
        for _ in range(self.sac_updates):
            state_value_loss, action_value_loss = self.update_value_functions(observations, actions)
        sac_time += timer.time() - ts

        # Step size computation
        n_step_size = 2.0 * self.kl_dist
        alpha = np.sqrt(np.abs(n_step_size / (np.dot(dapg_grad.T, npg_grad) + 1e-20)))

        # Policy update
        curr_params = self.policy.get_param_values()
        new_params = curr_params + alpha * npg_grad
        self.policy.set_param_values(new_params, set_new=True, set_old=False)

        surr_after = self.CPI_surrogate(observations, actions, advantages).data.numpy().ravel()[0]
        kl_dist = self.kl_old_new(observations, actions).data.numpy().ravel()[0]
        self.policy.set_param_values(new_params, set_new=True, set_old=True)

        # Log information
        if self.save_logs:
            self.logger.log_kv('alpha', alpha)
            self.logger.log_kv('delta', n_step_size)
            self.logger.log_kv('time_vpg', t_gLL)
            self.logger.log_kv('time_npg', t_FIM)
            self.logger.log_kv('time_SAC', sac_time)
            self.logger.log_kv('kl_dist', kl_dist)
            self.logger.log_kv('surr_improvement', surr_after - surr_before)
            self.logger.log_kv('running_score', self.running_score)

        try:
            self.env.env.env.evaluate_success(paths, self.logger)
        except:
            try:
                success_rate = self.env.env.env.evaluate_success(paths)
                self.logger.log_kv('success_rate', success_rate)
            except:
                pass

        return base_stats

    def update_value_functions(self, observations, actions):
        states = torch.from_numpy(observations).float()
        actions = torch.from_numpy(actions).float()
        next_states = torch.from_numpy(np.roll(observations, -1, axis=0)).float()
        dones = torch.from_numpy(np.zeros_like(states[:, 0])).float()

        with torch.no_grad():
            next_actions, next_log_probs = self.policy.sample(next_states)
            q_next_target = self.target_action_value_function(torch.cat([next_states, next_actions], dim=-1))
            v_next_target = self.target_state_value_function(next_states)
            target_q = self.policy.reward_function(states, actions) + self.env.gamma * (v_next_target - self.sac_temperature * next_log_probs)

        q = self.action_value_function(torch.cat([states, actions], dim=-1))
        v = self.state_value_function(states)

        value_loss = nn.MSELoss()(v, target_q.detach())
        action_value_loss = nn.MSELoss()(q, target_q)

        self.value_function_optimizer.zero_grad()
        (value_loss + action_value_loss).backward()
        self.value_function_optimizer.step()

        return value_loss.item(), action_value_loss.item()

    def soft_update_target_networks(self, tau):
        for target_param, param in zip(self.target_state_value_function.parameters(), self.state_value_function.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for target_param, param in zip(self.target_action_value_function.parameters(), self.action_value_function.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)