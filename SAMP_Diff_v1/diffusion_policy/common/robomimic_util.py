import copy

import h5py
import numpy as np
from scipy.spatial.transform import Rotation


class RobomimicAbsoluteActionConverter:
    """Convert robosuite delta EEF actions to equivalent absolute goals.

    Conversion must be performed through the controller at every recorded
    simulator state. Cumulative sums of delta actions are not equivalent to
    robosuite's scaled, frame-aware controller goals.
    """

    def __init__(self, dataset_path, algo_name="bc", create_abs_env=False):
        # Importing these packages registers their custom robosuite tasks.
        # Plain RoboMimic installations do not provide either package, so the
        # imports intentionally remain optional.
        for module_name in ("robosuite_task_zoo", "mimicgen", "mimicgen.envs"):
            try:
                __import__(module_name)
            except ImportError:
                pass

        import robomimic.utils.env_utils as EnvUtils
        import robomimic.utils.file_utils as FileUtils
        import robomimic.utils.obs_utils as ObsUtils
        from robomimic.config import config_factory

        config = config_factory(algo_name=algo_name)
        ObsUtils.initialize_obs_utils_with_config(config)

        self.dataset_path = str(dataset_path)
        self.file = h5py.File(self.dataset_path, "r")
        self.demo_keys = sorted(
            self.file["data"].keys(),
            key=lambda value: int(value.rsplit("_", 1)[-1]),
        )

        env_meta = FileUtils.get_env_metadata_from_dataset(self.dataset_path)
        controller = env_meta.get("env_kwargs", {}).get("controller_configs", {})
        if isinstance(controller, dict) and controller.get("control_delta") is False:
            raise ValueError(
                "dataset metadata already declares control_delta=False; "
                "refusing to convert an absolute-action dataset again"
            )

        self.env = EnvUtils.create_env_from_metadata(
            env_meta=env_meta,
            render=False,
            render_offscreen=False,
            use_image_obs=False,
        )
        if len(self.env.env.robots) not in (1, 2):
            raise ValueError("only one-arm and two-arm robosuite datasets are supported")

        self.abs_env = None
        if create_abs_env:
            abs_env_meta = copy.deepcopy(env_meta)
            abs_env_meta["env_kwargs"]["controller_configs"]["control_delta"] = False
            self.abs_env = EnvUtils.create_env_from_metadata(
                env_meta=abs_env_meta,
                render=False,
                render_offscreen=False,
                use_image_obs=False,
            )

    def __len__(self):
        return len(self.demo_keys)

    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None

    def convert_actions(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        if actions.shape[-1] not in (7, 14):
            raise ValueError(
                f"expected 7D or 14D robosuite actions, got {actions.shape}"
            )
        if len(states) != len(actions):
            raise ValueError(
                f"states/actions length mismatch: {len(states)} != {len(actions)}"
            )

        stacked_actions = actions.reshape(*actions.shape[:-1], -1, 7)
        goal_pos = np.zeros(stacked_actions.shape[:-1] + (3,), dtype=actions.dtype)
        goal_ori = np.zeros(stacked_actions.shape[:-1] + (3,), dtype=actions.dtype)
        gripper = stacked_actions[..., [-1]]

        for step, state in enumerate(states):
            self.env.reset_to({"states": state})
            for robot_idx, robot in enumerate(self.env.env.robots):
                robot.control(stacked_actions[step, robot_idx], policy_step=True)
                controller = robot.controller
                goal_pos[step, robot_idx] = controller.goal_pos
                goal_ori[step, robot_idx] = Rotation.from_matrix(
                    controller.goal_ori
                ).as_rotvec()

        abs_actions = np.concatenate((goal_pos, goal_ori, gripper), axis=-1)
        return abs_actions.reshape(actions.shape).astype(actions.dtype, copy=False)

    def convert_key(self, demo_key):
        demo = self.file[f"data/{demo_key}"]
        return self.convert_actions(demo["states"][:], demo["actions"][:])

    def convert_idx(self, idx):
        return self.convert_key(self.demo_keys[idx])

    @staticmethod
    def evaluate_rollout_error(
        env,
        states,
        actions,
        reference_eef_pos,
        reference_eef_quat,
        metric_skip_steps=1,
    ):
        rollout_pos = []
        rollout_quat = []
        for state, action in zip(states, actions):
            env.reset_to({"states": state})
            env.step(action)
            obs = env.get_observation()
            rollout_pos.append(obs["robot0_eef_pos"])
            rollout_quat.append(obs["robot0_eef_quat"])

        rollout_pos = np.asarray(rollout_pos)
        rollout_quat = np.asarray(rollout_quat)
        pos_dist = np.linalg.norm(reference_eef_pos[1:] - rollout_pos[:-1], axis=-1)
        rot_diff = Rotation.from_quat(reference_eef_quat[1:]) * Rotation.from_quat(
            rollout_quat[:-1]
        ).inv()
        start = min(int(metric_skip_steps), len(pos_dist))
        if start == len(pos_dist):
            return {"pos": float("nan"), "rot": float("nan")}
        return {
            "pos": float(np.max(pos_dist[start:])),
            "rot": float(np.max(rot_diff.magnitude()[start:])),
        }
    def convert_and_evaluate_key(self, demo_key):
        if self.abs_env is None:
            raise RuntimeError("create converter with create_abs_env=True for validation")
        demo = self.file[f"data/{demo_key}"]
        states = demo["states"][:]
        delta_actions = demo["actions"][:]
        abs_actions = self.convert_actions(states, delta_actions)
        pos = demo["obs/robot0_eef_pos"][:]
        quat = demo["obs/robot0_eef_quat"][:]
        return abs_actions, {
            "delta": self.evaluate_rollout_error(
                self.env, states, delta_actions, pos, quat
            ),
            "abs": self.evaluate_rollout_error(
                self.abs_env, states, abs_actions, pos, quat
            ),
        }
