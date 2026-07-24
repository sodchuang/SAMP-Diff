from typing import List, Dict, Optional, TYPE_CHECKING
import numpy as np
import gym
from gym.spaces import Box

if TYPE_CHECKING:
    from robomimic.envs.env_robosuite import EnvRobosuite
else:
    EnvRobosuite = object

class RobomimicLowdimWrapper(gym.Env):
    def __init__(self, 
        env: EnvRobosuite,
        obs_keys: List[str]=[
            'object', 
            'robot0_eef_pos', 
            'robot0_eef_quat', 
            'robot0_gripper_qpos'],
        init_state: Optional[np.ndarray]=None,
        render_hw=(256,256),
        render_camera_name='agentview',
        tool_hang_lift_delta=0.03,
        tool_hang_hold_steps=10,
        tool_hang_insert_hold_steps=10,
        tool_hang_full_hold_steps=10,
        ):

        self.env = env
        self.obs_keys = obs_keys
        self.init_state = init_state
        self.render_hw = render_hw
        self.render_camera_name = render_camera_name
        self.seed_state_map = dict()
        self._seed = None
        self.tool_hang_lift_delta = float(tool_hang_lift_delta)
        self.tool_hang_hold_steps = int(tool_hang_hold_steps)
        self.tool_hang_insert_hold_steps = int(tool_hang_insert_hold_steps)
        self.tool_hang_full_hold_steps = int(tool_hang_full_hold_steps)
        self._reset_tool_hang_stage_state()
        
        # setup spaces
        low = np.full(env.action_dimension, fill_value=-1)
        high = np.full(env.action_dimension, fill_value=1)
        self.action_space = Box(
            low=low,
            high=high,
            shape=low.shape,
            dtype=low.dtype
        )
        obs_example = self.get_observation()
        low = np.full_like(obs_example, fill_value=-1)
        high = np.full_like(obs_example, fill_value=1)
        self.observation_space = Box(
            low=low,
            high=high,
            shape=low.shape,
            dtype=low.dtype
        )

    def get_observation(self):
        raw_obs = self.env.get_observation()
        obs = np.concatenate([
            raw_obs[key] for key in self.obs_keys
        ], axis=0)
        return obs

    def seed(self, seed=None):
        np.random.seed(seed=seed)
        self._seed = seed
    
    def reset(self):
        if self.init_state is not None:
            # always reset to the same state
            # to be compatible with gym
            self.env.reset_to({'states': self.init_state})
        elif self._seed is not None:
            # reset to a specific seed
            seed = self._seed
            if seed in self.seed_state_map:
                # env.reset is expensive, use cache
                self.env.reset_to({'states': self.seed_state_map[seed]})
            else:
                # robosuite's initializes all use numpy global random state
                np.random.seed(seed=seed)
                self.env.reset()
                state = self.env.get_state()['states']
                self.seed_state_map[seed] = state
            self._seed = None
        else:
            # random reset
            self.env.reset()

        self._reset_tool_hang_stage_state()
        self._tool_hang_initial_frame_z = self._tool_hang_frame_z()

        # return obs
        obs = self.get_observation()
        return obs
    
    def step(self, action):
        raw_obs, reward, done, info = self.env.step(action)
        self._update_tool_hang_stage_state()
        obs = np.concatenate([
            raw_obs[key] for key in self.obs_keys
        ], axis=0)
        return obs, reward, done, info

    def _tool_hang_task(self):
        """Return the underlying robosuite ToolHang task, if this is one."""
        task = getattr(self.env, 'env', None)
        required = ('_check_grasp', '_check_frame_assembled', '_check_success')
        if task is None or not all(hasattr(task, name) for name in required):
            return None
        if not hasattr(task, 'frame') or not hasattr(task, 'robots'):
            return None
        return task

    def _tool_hang_frame_z(self):
        task = self._tool_hang_task()
        if task is None:
            return None
        try:
            body_id = task.sim.model.body_name2id(task.frame.root_body)
            return float(task.sim.data.body_xpos[body_id][2])
        except Exception:
            return None

    def _tool_hang_frame_grasped(self, task):
        try:
            gripper = task.robots[0].gripper
            geoms = task.frame.contact_geoms
            try:
                return bool(task._check_grasp(
                    gripper=gripper, object_geoms=geoms))
            except TypeError:
                return bool(task._check_grasp(gripper, geoms))
        except Exception:
            return False

    def _reset_tool_hang_stage_state(self):
        self._tool_hang_initial_frame_z = None
        self._tool_hang_consecutive = {'grasp': 0, 'insert': 0, 'full': 0}
        self._tool_hang_max_consecutive = {'grasp': 0, 'insert': 0, 'full': 0}
        self._tool_hang_succeeded = {
            'grasp': False, 'insert': False, 'full': False}
        self._tool_hang_seen_frame_grasp = False
        self._tool_hang_seen_frame_lift = False
        self._tool_hang_max_frame_lift_delta = 0.0

    def _update_tool_hang_stage_state(self):
        task = self._tool_hang_task()
        if task is None:
            return

        frame_z = self._tool_hang_frame_z()
        if self._tool_hang_initial_frame_z is None and frame_z is not None:
            self._tool_hang_initial_frame_z = frame_z

        grasped = self._tool_hang_frame_grasped(task)
        lifted = (
            frame_z is not None
            and self._tool_hang_initial_frame_z is not None
            and frame_z >= self._tool_hang_initial_frame_z + self.tool_hang_lift_delta
        )
        lift_delta = 0.0
        if frame_z is not None and self._tool_hang_initial_frame_z is not None:
            lift_delta = frame_z - self._tool_hang_initial_frame_z
        self._tool_hang_seen_frame_grasp |= bool(grasped)
        self._tool_hang_seen_frame_lift |= bool(lifted)
        self._tool_hang_max_frame_lift_delta = max(
            self._tool_hang_max_frame_lift_delta, float(lift_delta))
        try:
            assembled = bool(task._check_frame_assembled())
        except Exception:
            assembled = False
        try:
            full = bool(task._check_success())
        except Exception:
            full = False

        raw = {
            'grasp': grasped and lifted,
            'insert': assembled and not grasped,
            'full': full,
        }
        thresholds = {
            'grasp': self.tool_hang_hold_steps,
            'insert': self.tool_hang_insert_hold_steps,
            'full': self.tool_hang_full_hold_steps,
        }
        for name, active in raw.items():
            self._tool_hang_consecutive[name] = (
                self._tool_hang_consecutive[name] + 1 if active else 0)
            self._tool_hang_max_consecutive[name] = max(
                self._tool_hang_max_consecutive[name],
                self._tool_hang_consecutive[name],
            )
            if self._tool_hang_consecutive[name] >= thresholds[name]:
                self._tool_hang_succeeded[name] = True

    def get_tool_hang_stage_flags(self):
        """Pickle-safe per-episode stage result used by AsyncVectorEnv."""
        return {
            'available': bool(self._tool_hang_task() is not None),
            'grasp': bool(self._tool_hang_succeeded['grasp']),
            'insert': bool(self._tool_hang_succeeded['insert']),
            'full': bool(self._tool_hang_succeeded['full']),
            'frame_grasp_contact': bool(self._tool_hang_seen_frame_grasp),
            'frame_lift': bool(self._tool_hang_seen_frame_lift),
            'max_frame_lift_delta': float(self._tool_hang_max_frame_lift_delta),
            'max_grasp_hold_steps': int(
                self._tool_hang_max_consecutive['grasp']),
            'grasp_consecutive_steps': int(self._tool_hang_consecutive['grasp']),
            'insert_consecutive_steps': int(self._tool_hang_consecutive['insert']),
            'full_consecutive_steps': int(self._tool_hang_consecutive['full']),
        }
    
    def render(self, mode='rgb_array'):
        h, w = self.render_hw
        return self.env.render(mode=mode, 
            height=h, width=w, 
            camera_name=self.render_camera_name)


def test():
    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.env_utils as EnvUtils
    from matplotlib import pyplot as plt

    dataset_path = '/home/cchi/dev/diffusion_policy/data/robomimic/datasets/square/ph/low_dim.hdf5'
    env_meta = FileUtils.get_env_metadata_from_dataset(
        dataset_path)

    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False, 
        render_offscreen=False,
        use_image_obs=False, 
    )
    wrapper = RobomimicLowdimWrapper(
        env=env,
        obs_keys=[
            'object', 
            'robot0_eef_pos', 
            'robot0_eef_quat', 
            'robot0_gripper_qpos'
        ]
    )

    states = list()
    for _ in range(2):
        wrapper.seed(0)
        wrapper.reset()
        states.append(wrapper.env.get_state()['states'])
    assert np.allclose(states[0], states[1])

    img = wrapper.render()
    plt.imshow(img)
    # wrapper.seed()
    # states.append(wrapper.env.get_state()['states'])
