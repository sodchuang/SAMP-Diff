import os
import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import dill
import math
import wandb.sdk.data_types.video as wv
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
# from diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
from diffusion_policy.model.common.rotation_transformer import RotationTransformer

from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
from diffusion_policy.env.robomimic.robomimic_lowdim_wrapper import RobomimicLowdimWrapper


def create_env(env_meta, obs_keys):
    env_name = str(env_meta.get('env_name', ''))
    # MimicGen environments register themselves with robosuite at import
    # time.  Without this import, dataset loading succeeds but the first
    # rollout fails much later with an unknown-environment error.
    if env_name in {'Stack_D1', 'NutAssembly_D0', 'Threading_D2'}:
        try:
            import mimicgen.envs.robosuite  # noqa: F401
        except ImportError as e:
            raise ImportError(
                f"Dataset requires MimicGen environment {env_name!r}, but "
                "mimicgen.envs.robosuite is not importable. Install the "
                "official MimicGen package in a compatible environment "
                "before starting training."
            ) from e
    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.obs_utils as ObsUtils

    ObsUtils.initialize_obs_modality_mapping_from_dict(
        {'low_dim': obs_keys})
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False, 
        # only way to not show collision geometry
        # is to enable render_offscreen
        # which uses a lot of RAM.
        render_offscreen=False,
        use_image_obs=False, 
    )
    return env


class RobomimicLowdimRunner(BaseLowdimRunner):
    """
    Robomimic envs already enforces number of steps.
    """

    def __init__(self, 
            output_dir,
            dataset_path,
            obs_keys,
            n_train=10,
            n_train_vis=3,
            train_start_idx=0,
            n_test=22,
            n_test_vis=6,
            test_start_seed=10000,
            max_steps=400,
            n_obs_steps=2,
            n_action_steps=8,
            n_latency_steps=0,
            render_hw=(256,256),
            render_camera_name='agentview',
            fps=10,
            crf=22,
            past_action=False,
            abs_action=False,
            action_clip_by_dataset=False,
            action_clip_margin_scale=0.25,
            action_clip_min_margin=0.05,
            tool_hang_lift_delta=0.03,
            tool_hang_hold_steps=10,
            tool_hang_insert_hold_steps=10,
            tool_hang_full_hold_steps=10,
            stack_release_hold_steps=1,
            stack_min_eef_distance=0.04,
            stack_min_gripper_open_width=0.04,
            tqdm_interval_sec=5.0,
            n_envs=None
        ):
        """
        Assuming:
        n_obs_steps=2
        n_latency_steps=3
        n_action_steps=4
        o: obs
        i: inference
        a: action
        Batch t:
        |o|o| | | | | | |
        | |i|i|i| | | | |
        | | | | |a|a|a|a|
        Batch t+1
        | | | | |o|o| | | | | | |
        | | | | | |i|i|i| | | | |
        | | | | | | | | |a|a|a|a|
        """

        super().__init__(output_dir)

        if n_envs is None:
            n_envs = n_train + n_test

        # handle latency step
        # to mimic latency, we request n_latency_steps additional steps 
        # of past observations, and the discard the last n_latency_steps
        env_n_obs_steps = n_obs_steps + n_latency_steps
        env_n_action_steps = n_action_steps

        # assert n_obs_steps <= n_action_steps
        dataset_path = os.path.expanduser(dataset_path)
        robosuite_fps = 20
        steps_per_render = max(robosuite_fps // fps, 1)

        try:
            import h5py
            import robomimic.utils.file_utils as FileUtils
        except ImportError as e:
            raise ImportError(
                "RobomimicLowdimRunner requires h5py and robomimic. "
                "Install them in the training environment before rollout."
            ) from e

        # read from dataset
        env_meta = FileUtils.get_env_metadata_from_dataset(
            dataset_path)
        rotation_transformer = None
        if abs_action:
            env_meta['env_kwargs']['controller_configs']['control_delta'] = False
            rotation_transformer = RotationTransformer('axis_angle', 'rotation_6d')

        self.action_clip_by_dataset = bool(action_clip_by_dataset)
        self.action_clip_min = None
        self.action_clip_max = None
        self._action_clip_warned = False
        if self.action_clip_by_dataset:
            action_min, action_max = self._get_dataset_action_bounds(
                dataset_path=dataset_path,
                margin_scale=action_clip_margin_scale,
                min_margin=action_clip_min_margin,
            )
            self.action_clip_min = action_min
            self.action_clip_max = action_max

        def env_fn():
            robomimic_env = create_env(
                    env_meta=env_meta, 
                    obs_keys=obs_keys
                )
            # hard reset doesn't influence lowdim env
            # robomimic_env.env.hard_reset = False
            return MultiStepWrapper(
                    VideoRecordingWrapper(
                        RobomimicLowdimWrapper(
                            env=robomimic_env,
                            obs_keys=obs_keys,
                            init_state=None,
                            render_hw=render_hw,
                            render_camera_name=render_camera_name,
                            tool_hang_lift_delta=tool_hang_lift_delta,
                            tool_hang_hold_steps=tool_hang_hold_steps,
                            tool_hang_insert_hold_steps=tool_hang_insert_hold_steps,
                            tool_hang_full_hold_steps=tool_hang_full_hold_steps,
                            stack_release_hold_steps=stack_release_hold_steps,
                            stack_min_eef_distance=stack_min_eef_distance,
                            stack_min_gripper_open_width=stack_min_gripper_open_width,
                        ),
                        video_recoder=VideoRecorder.create_h264(
                            fps=fps,
                            codec='h264',
                            input_pix_fmt='rgb24',
                            crf=crf,
                            thread_type='FRAME',
                            thread_count=1
                        ),
                        file_path=None,
                        steps_per_render=steps_per_render
                    ),
                    n_obs_steps=env_n_obs_steps,
                    n_action_steps=env_n_action_steps,
                    max_episode_steps=max_steps
                )

        env_fns = [env_fn] * n_envs
        env_seeds = list()
        env_prefixs = list()
        env_init_fn_dills = list()

        # train
        with h5py.File(dataset_path, 'r') as f:
            for i in range(n_train):
                train_idx = train_start_idx + i
                enable_render = i < n_train_vis
                init_state = f[f'data/demo_{train_idx}/states'][0]

                def init_fn(env, init_state=init_state, 
                    enable_render=enable_render):
                    # setup rendering
                    # video_wrapper
                    assert isinstance(env.env, VideoRecordingWrapper)
                    env.env.video_recoder.stop()
                    env.env.file_path = None
                    if enable_render:
                        filename = pathlib.Path(output_dir).joinpath(
                            'media', wv.util.generate_id() + ".mp4")
                        filename.parent.mkdir(parents=False, exist_ok=True)
                        filename = str(filename)
                        env.env.file_path = filename

                    # switch to init_state reset
                    assert isinstance(env.env.env, RobomimicLowdimWrapper)
                    env.env.env.init_state = init_state

                env_seeds.append(train_idx)
                env_prefixs.append('train/')
                env_init_fn_dills.append(dill.dumps(init_fn))
        
        # test
        for i in range(n_test):
            seed = test_start_seed + i
            enable_render = i < n_test_vis

            def init_fn(env, seed=seed, 
                enable_render=enable_render):
                # setup rendering
                # video_wrapper
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        'media', wv.util.generate_id() + ".mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                # switch to seed reset
                assert isinstance(env.env.env, RobomimicLowdimWrapper)
                env.env.env.init_state = None
                env.seed(seed)

            env_seeds.append(seed)
            env_prefixs.append('test/')
            env_init_fn_dills.append(dill.dumps(init_fn))
        
        env = AsyncVectorEnv(env_fns)
        # env = SyncVectorEnv(env_fns)

        self.env_meta = env_meta
        normalized_env_name = ''.join(
            char for char in str(env_meta.get('env_name', '')).lower()
            if char.isalnum()
        )
        self.is_tool_hang = 'toolhang' in normalized_env_name
        self.is_stack = normalized_env_name.startswith('stack')
        self.env = env
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.n_latency_steps = n_latency_steps
        self.env_n_obs_steps = env_n_obs_steps
        self.env_n_action_steps = env_n_action_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.rotation_transformer = rotation_transformer
        self.abs_action = abs_action
        self.tqdm_interval_sec = tqdm_interval_sec

    def recover_after_worker_error(self):
        """Replace a broken AsyncVectorEnv after a simulator worker exits.

        MuJoCo warnings such as an unstable QACC terminate the affected worker.
        That vector environment cannot be reused, but training can safely
        continue after all remaining workers are terminated and recreated.
        """
        old_env = self.env
        try:
            old_env.close_extras(terminate=True)
        except Exception as exc:
            print(f"[WARN] Failed to fully close broken rollout workers: {exc}")
        self.env = AsyncVectorEnv(self.env_fns)
        print("[WARN] Recreated rollout environments after worker failure.")

    @staticmethod
    def _get_dataset_action_bounds(dataset_path, margin_scale=0.25, min_margin=0.05):
        import h5py

        mins = list()
        maxs = list()
        with h5py.File(dataset_path, 'r') as f:
            for demo in f['data'].values():
                actions = demo['actions'][:].astype(np.float32)
                mins.append(np.min(actions, axis=0))
                maxs.append(np.max(actions, axis=0))

        action_min = np.min(np.stack(mins, axis=0), axis=0)
        action_max = np.max(np.stack(maxs, axis=0), axis=0)
        action_range = action_max - action_min
        margin = np.maximum(action_range * float(margin_scale), float(min_margin))
        return action_min - margin, action_max + margin

    def _sanitize_env_action(self, env_action):
        if not np.all(np.isfinite(env_action)):
            if self.action_clip_min is None:
                print(env_action)
                raise RuntimeError("Nan or Inf env action")
            env_action = np.nan_to_num(
                env_action,
                nan=0.0,
                posinf=np.max(self.action_clip_max),
                neginf=np.min(self.action_clip_min),
            )

        if self.action_clip_min is None:
            return env_action

        clipped = np.clip(env_action, self.action_clip_min, self.action_clip_max)
        if (not self._action_clip_warned) and np.any(clipped != env_action):
            before_min = np.min(env_action, axis=(0, 1))
            before_max = np.max(env_action, axis=(0, 1))
            print(
                "[WARN] RobomimicLowdimRunner clipped env action outside "
                f"dataset bounds. min={before_min}, max={before_max}"
            )
            self._action_clip_warned = True
        return clipped

    def run(self, policy: BaseLowdimPolicy):
        device = policy.device
        dtype = policy.dtype
        env = self.env
        
        # plan for rollout
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        # allocate data
        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits
        all_stage_flags = [None] * n_inits
        all_stack_flags = [None] * n_inits

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0,this_n_active_envs)
            
            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]]*n_diff)
            assert len(this_init_fns) == n_envs

            # init envs
            env.call_each('run_dill_function', 
                args_list=[(x,) for x in this_init_fns])

            # start rollout
            obs = env.reset()
            past_action = None
            policy.reset()

            env_name = self.env_meta['env_name']
            pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval {env_name}Lowdim {chunk_idx+1}/{n_chunks}", 
                leave=False, mininterval=self.tqdm_interval_sec)

            done = False
            while not done:
                # create obs dict
                np_obs_dict = {
                    # handle n_latency_steps by discarding the last n_latency_steps
                    'obs': obs[:,:self.n_obs_steps].astype(np.float32)
                }
                if self.past_action and (past_action is not None):
                    # TODO: not tested
                    np_obs_dict['past_action'] = past_action[
                        :,-(self.n_obs_steps-1):].astype(np.float32)
                
                # device transfer
                obs_dict = dict_apply(np_obs_dict, 
                    lambda x: torch.from_numpy(x).to(
                        device=device))

                # run policy
                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)

                # device_transfer
                np_action_dict = dict_apply(action_dict,
                    lambda x: x.detach().to('cpu').numpy())

                # handle latency_steps, we discard the first n_latency_steps actions
                # to simulate latency
                action = np_action_dict['action'][:,self.n_latency_steps:]
                if not np.all(np.isfinite(action)):
                    print(action)
                    raise RuntimeError("Nan or Inf action")
                
                # step env
                env_action = action
                if self.abs_action:
                    env_action = self.undo_transform_action(action)
                env_action = self._sanitize_env_action(env_action)

                obs, reward, done, info = env.step(env_action)
                done = np.all(done)
                past_action = action

                # update pbar
                pbar.update(action.shape[1])
            pbar.close()

            # collect data for this round
            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]
            if self.is_tool_hang:
                all_stage_flags[this_global_slice] = env.call(
                    'get_tool_hang_stage_flags')[this_local_slice]
            if self.is_stack:
                all_stack_flags[this_global_slice] = env.call(
                    'get_stack_release_flags')[this_local_slice]

        # log
        max_rewards = collections.defaultdict(list)
        final_rewards = collections.defaultdict(list)
        stable_rewards = collections.defaultdict(list)
        stage_results = {
            'grasp': collections.defaultdict(list),
            'insert': collections.defaultdict(list),
            'full': collections.defaultdict(list),
        }
        stage_diagnostics = {
            'frame_grasp_contact': collections.defaultdict(list),
            'frame_lift': collections.defaultdict(list),
            'max_frame_lift_delta': collections.defaultdict(list),
            'max_grasp_hold_steps': collections.defaultdict(list),
        }
        log_data = dict()
        unavailable_stage_flags = [
            i for i, flags in enumerate(all_stage_flags)
            if flags is None or not bool(flags.get('available', False))
        ]
        if self.is_tool_hang and unavailable_stage_flags:
            raise RuntimeError(
                "ToolHang simulator-state metrics are unavailable for "
                f"{len(unavailable_stage_flags)}/{n_inits} rollouts. "
                "Check robosuite ToolHang compatibility and ensure the "
                "ToolHang wrapper / runner files were synced together."
            )
        unavailable_stack_flags = [
            i for i, flags in enumerate(all_stack_flags)
            if flags is None or not bool(flags.get('available', False))
        ]
        if self.is_stack and unavailable_stack_flags:
            raise RuntimeError(
                "Stack simulator-state metrics are unavailable for "
                f"{len(unavailable_stack_flags)}/{n_inits} rollouts. "
                "Ensure the Stack wrapper and runner files were synced together."
            )
        # results reported in the paper are generated using the commented out line below
        # which will only report and average metrics from first n_envs initial condition and seeds
        # fortunately this won't invalidate our conclusion since
        # 1. This bug only affects the variance of metrics, not their mean
        # 2. All baseline methods are evaluated using the same code
        # to completely reproduce reported numbers, uncomment this line:
        # for i in range(len(self.env_fns)):
        # and comment out this line
        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            reward_history = np.asarray(all_rewards[i], dtype=np.float32)
            max_reward = float(np.max(reward_history))
            final_reward = float(reward_history[-1])
            stable_window = reward_history[-10:]
            stable_reward = float(np.all(stable_window >= 1.0))

            stack_flags = all_stack_flags[i]
            if self.is_stack:
                # Do not count transient contact as Stack success. The main
                # paper metric is 1 only when the cube remains released on the
                # target and the gripper has moved away at the episode end.
                raw_max_reward = max_reward
                strict_score = float(bool(stack_flags.get(
                    'strict_success', False)))
                max_reward = strict_score
                final_reward = strict_score
                stable_reward = strict_score
                log_data[prefix+f'sim_raw_max_reward_{seed}'] = raw_max_reward
                log_data[prefix+f'stack_release_success_{seed}'] = strict_score
                for key in (
                    'final_released_stacked', 'final_gripper_away',
                    'final_gripper_open',
                    'seen_released_stacked', 'seen_gripper_away',
                    'final_eef_distance', 'final_gripper_open_width',
                    'release_consecutive_steps',
                    'max_release_consecutive_steps',
                ):
                    log_data[prefix+f'stack_{key}_{seed}'] = float(
                        stack_flags.get(key, 0.0))
            max_rewards[prefix].append(max_reward)
            final_rewards[prefix].append(final_reward)
            stable_rewards[prefix].append(stable_reward)
            log_data[prefix+f'sim_max_reward_{seed}'] = max_reward
            log_data[prefix+f'sim_final_reward_{seed}'] = final_reward
            log_data[prefix+f'sim_stable_reward_{seed}'] = stable_reward

            flags = all_stage_flags[i]
            if flags is not None and bool(flags.get('available', False)):
                for stage_name in stage_results:
                    succeeded = float(bool(flags.get(stage_name, False)))
                    stage_results[stage_name][prefix].append(succeeded)
                    log_data[
                        prefix+f'stage_{stage_name}_success_{seed}'
                    ] = succeeded
                for diagnostic_name in stage_diagnostics:
                    value = float(flags.get(diagnostic_name, 0.0))
                    stage_diagnostics[diagnostic_name][prefix].append(value)
                    log_data[
                        prefix+f'{diagnostic_name}_{seed}'
                    ] = value

            # visualize sim
            video_path = all_video_paths[i]
            if video_path is not None:
                sim_video = wandb.Video(video_path)
                log_data[prefix+f'sim_video_{seed}'] = sim_video

        # log aggregate metrics
        for prefix, value in max_rewards.items():
            name = prefix+'mean_score'
            value = float(np.mean(value))
            log_data[name] = value
        for prefix, value in final_rewards.items():
            name = prefix+'mean_final_score'
            value = float(np.mean(value))
            log_data[name] = value
        for prefix, value in stable_rewards.items():
            name = prefix+'mean_stable_score'
            value = float(np.mean(value))
            log_data[name] = value
        if self.is_stack:
            for prefix in ('train/', 'test/'):
                values = [
                    value for key, value in log_data.items()
                    if key.startswith(prefix+'stack_release_success_')
                ]
                if values:
                    log_data[prefix+'stack_release_rate'] = float(
                        np.mean(values))
        for stage_name, prefix_values in stage_results.items():
            for prefix, value in prefix_values.items():
                log_data[prefix+f'stage_{stage_name}_rate'] = float(
                    np.mean(value))
        for diagnostic_name, prefix_values in stage_diagnostics.items():
            for prefix, value in prefix_values.items():
                if diagnostic_name in ('frame_grasp_contact', 'frame_lift'):
                    metric_name = prefix+f'{diagnostic_name}_rate'
                else:
                    metric_name = prefix+f'mean_{diagnostic_name}'
                log_data[metric_name] = float(np.mean(value))

        return log_data

    def undo_transform_action(self, action):
        raw_shape = action.shape
        if raw_shape[-1] == 20:
            # dual arm
            action = action.reshape(-1,2,10)

        d_rot = action.shape[-1] - 4
        pos = action[...,:3]
        rot = action[...,3:3+d_rot]
        gripper = action[...,[-1]]
        rot = self.rotation_transformer.inverse(rot)
        uaction = np.concatenate([
            pos, rot, gripper
        ], axis=-1)

        if raw_shape[-1] == 20:
            # dual arm
            uaction = uaction.reshape(*raw_shape[:-1], 14)

        return uaction
