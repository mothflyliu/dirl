import os

from gym import error, spaces
from gym.utils import seeding
import numpy as np
from os import path
import gym
import six
import time as timer

try:
    import mujoco_py
    from mujoco_py import load_model_from_path, MjSim, MjViewer
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))

def get_sim(model_path):
    if model_path.startswith("/"):
        fullpath = model_path
    else:
        fullpath = os.path.join(os.path.dirname(__file__), "assets", model_path)
    if not path.exists(fullpath):
        raise IOError("File %s does not exist" % fullpath)
    model = load_model_from_path(fullpath)
    return MjSim(model)

class MujocoEnv(gym.Env):
    """Superclass for all MuJoCo environments.
    """

    def __init__(self, model_path=None, frame_skip=1, sim=None):
        # 如果没有提供模拟对象
        if sim is None:
            # 获取模拟对象
            self.sim = get_sim(model_path)
        else:
            # 使用提供的模拟对象
            self.sim = sim
        # 获取模拟数据
        self.data = self.sim.data
        # 获取模拟模型
        self.model = self.sim.model
        # 设置帧跳过值
        self.frame_skip = frame_skip
        # 设置元数据
        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }
        # 默认为不渲染 mujoco 帧
        self.mujoco_render_frames = False
        # 保存初始位置状态
        self.init_qpos = self.data.qpos.ravel().copy()
        # 保存初始速度状态
        self.init_qvel = self.data.qvel.ravel().copy()
        # 尝试执行一步操作
        try:
            observation, _reward, done, _info = self.step(np.zeros(self.model.nu))
        except NotImplementedError:
            observation, _reward, done, _info = self._step(np.zeros(self.model.nu))
        # 断言操作未完成
        assert not done
        # 获取观察维度
        self.obs_dim = np.sum([o.size for o in observation]) if type(observation) is tuple else observation.size
        # 获取动作边界
        bounds = self.model.actuator_ctrlrange.copy()
        # 下限
        low = bounds[:, 0]
        # 上限
        high = bounds[:, 1]
        # 设置动作空间
        self.action_space = spaces.Box(low, high, dtype=np.float32)
        # 上限为正无穷
        high = np.inf*np.ones(self.obs_dim)
        # 下限为负无穷
        low = -high
        # 设置观察空间
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        self.seed()

    def seed(self, seed=None):
        # 设置随机数生成器和种子
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # methods to override:
    # ----------------------------

    def reset_model(self):
        """
        Reset the robot degrees of freedom (qpos and qvel).
        Implement this in each subclass.
        """
        raise NotImplementedError

    def mj_viewer_setup(self):
        """
        Due to specifics of new mujoco rendering, the standard viewer cannot be used
        with this set-up. Instead we use this mujoco specific function.
        """
        pass

    def viewer_setup(self):
        """
        Does not work. Use mj_viewer_setup() instead
        """
        pass

    def evaluate_success(self, paths, logger=None):
        """
        Log various success metrics calculated based on input paths into the logger
        """
        pass

    # -----------------------------

    def reset(self):
        self.sim.reset()
        self.sim.forward()
        ob = self.reset_model()
        return ob

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel,
                                         old_state.act, old_state.udd_state)
        self.sim.set_state(new_state)
        self.sim.forward()

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    def do_simulation(self, ctrl, n_frames):
        # 对于每个控制维度
        for i in range(self.model.nu):
            # 将控制输入值设置到模拟数据的控制数组中
            self.sim.data.ctrl[i] = ctrl[i]
        # 执行指定帧数的模拟步骤
        for _ in range(n_frames):
            # 执行模拟的一步
            self.sim.step()
            # 如果设置了渲染帧
            if self.mujoco_render_frames is True:
                # 进行渲染
                self.mj_render()

    def mj_render(self):
        try:
            self.viewer.render()
        except:
            self.mj_viewer_setup()
            self.viewer._run_speed = 0.5
            #self.viewer._run_speed /= self.frame_skip
            self.viewer.render()

    def render(self, *args, **kwargs):
        pass
        #return self.mj_render()

    def _get_viewer(self):
        pass
        #return None

    def state_vector(self):
        state = self.sim.get_state()
        return np.concatenate([
            state.qpos.flat, state.qvel.flat])

    # -----------------------------

    def visualize_policy(self, policy, horizon=1000, num_episodes=1, mode='exploration'):
        self.mujoco_render_frames = True
        for ep in range(num_episodes):
            o = self.reset()
            d = False
            t = 0
            score = 0.0
            while t < horizon and d is False:
                a = policy.get_action(o)[0] if mode == 'exploration' else policy.get_action(o)[1]['evaluation']
                o, r, d, _ = self.step(a)
                t = t+1
                score = score + r
            print("Episode score = %f" % score)
        self.mujoco_render_frames = False

    def visualize_policy_offscreen(self, policy, horizon=1000,
                                   num_episodes=1,
                                   frame_size=(640,480),
                                   mode='exploration',
                                   save_loc='/tmp/',
                                   filename='newvid',
                                   camera_name=None):
        import skvideo.io
        for ep in range(num_episodes):
            print("Episode %d: rendering offline " % ep, end='', flush=True)
            o = self.reset()
            d = False
            t = 0
            arrs = []
            t0 = timer.time()
            while t < horizon and d is False:
                a = policy.get_action(o)[0] if mode == 'exploration' else policy.get_action(o)[1]['evaluation']
                o, r, d, _ = self.step(a)
                t = t+1
                curr_frame = self.sim.render(width=frame_size[0], height=frame_size[1],
                                             mode='offscreen', camera_name=camera_name, device_id=0)
                arrs.append(curr_frame[::-1,:,:])
                print(t, end=', ', flush=True)
            file_name = save_loc + filename + str(ep) + ".mp4"
            skvideo.io.vwrite( file_name, np.asarray(arrs))
            print("saved", file_name)
            t1 = timer.time()
            print("time taken = %f"% (t1-t0))
