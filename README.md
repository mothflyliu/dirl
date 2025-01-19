# Supervised Optimal Control in Complex Continuous Systems with Trajectory Imitation and Reinforcement Learning

This package  contains implementations of various RL algorithms for continuous control tasks simulated with [MuJoCo.](http://www.mujoco.org/). A novel supervised optimal control framework based on trajectory imitation (TI) and reinforcement learning (RL) for complex continuous systems is proposed in this paper. Firstly, behavior cloning (BC) is used to pre-train the policy model through a small number of human demonstrations, which learns a primary policy through supervised learning to mimic the demonstrations. Secondly, a generative adversarial imitation learning (GAIL) method is carried out to learn the implicit characteristics of demonstration data.

# Installation
The main package dependencies are `MuJoCo`, `python=3.7`, `gym>=0.13`, `mujoco-py>=2.0`, and `pytorch>=1.0`. See `setup/README.md` for detailed install instructions.

# Getting started
Each repository above contains detailed setup instructions. 
1. **Step 1:** Install [mjrl], using instructions in the repository. `mjrl` comes with an anaconda environment which helps to easily import and use a variety of MuJoCo tasks.
2. **Step 2:** Install [mj_envs] by following the instructions in the repository. Note that `mj_envs` uses git submodules, and hence must be cloned correctly per instructions in the repo.
3. **Step 3:** After setting up `mjrl` and `mj_envs`, clone this repository and use the following commands to visualize the demonstrations and pre-trained policies.
4. **Step 4:**
run_bgdnpg_last.py, run this for proposed TIRL method for the optimal control algorithm based on trajectory learning & reinforcement learning.
run_dapg.py, run this for DAPG method for the optimal control algorithm.
run_npg.py, run this for NPG method for the optimal control algorithm based on trajectory learning & reinforcement learning.
run_bcrl.py, run this for BCNPG method for the optimal control algorithm based on trajectory learning & reinforcement learning.

#  Workspace description
Perform different tasks with dexterous hands under a centralized algorithm is a challenging problem, especially for complex continuous systems. The experimental platform is used to to accomplish the validation tasks of our methods on optimal control of complex continuous system. It's a simulated analogue of a highly dexterous manipulator with 24-DoF and each DoF is actuated using position control equipped with a joint angle sensor.
