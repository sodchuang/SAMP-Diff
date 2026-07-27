if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import numpy as np
import random
import wandb
import tqdm

from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.samp_lowdim_policy import SampLowdimPolicy
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusion_policy.model.diffusion.ema_model import EMAModel

OmegaConf.register_new_resolver("eval", eval, replace=True)


def _apply_gripper_close_semantics(policy, threshold, close_is_greater):
    """Keep dataset event detection, training loss and rollout latch aligned."""
    threshold = float(threshold)
    close_is_greater = bool(close_is_greater)
    timing_cfg = getattr(policy, 'action_group_timing_params', None)
    if timing_cfg and timing_cfg.get('gripper', None):
        timing_cfg['gripper']['close_threshold'] = threshold
        timing_cfg['gripper']['close_is_greater'] = close_is_greater
    phase_cfg = getattr(policy.samp_net, 'action_phase_loss_params', None)
    if phase_cfg:
        phase_cfg['close_threshold'] = threshold
        phase_cfg['close_is_greater'] = close_is_greater
    print(
        "[Dataset] applied gripper close semantics to policy: "
        f"threshold={threshold:.6g}, close_is_greater={close_is_greater}"
    )


def _set_samp_encoder_trainable(policy, trainable):
    """Freeze the observation / trajectory encoder during early transfer."""
    prefixes = (
        'samp_net.condition_proj.',
        'samp_net.embedding_index.',
        'samp_net.z_proj.',
        'samp_net.z_proj_ln.',
        'samp_net.encoder_blocks.',
        'samp_net.encoder_norm.',
        'samp_net.encoder_pos_embed_learned',
    )
    changed = False
    for name, parameter in policy.named_parameters():
        if name.startswith(prefixes) and parameter.requires_grad != trainable:
            parameter.requires_grad_(trainable)
            changed = True
    return changed


def _is_mujoco_instability_error(exc):
    message = str(exc).lower()
    return (
        "mujoco" in message
        and (
            "simulation is unstable" in message
            or "huge value in qacc" in message
            or "nan, inf or huge value" in message
        )
    )


class TrainSampLowdimWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: SampLowdimPolicy
        self.model = hydra.utils.instantiate(cfg.policy)

        self.ema_model: SampLowdimPolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters()
        )

        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # resume training
        resume_from_path = cfg.training.get('resume_from_path', None)
        cross_stage_transfer = bool(resume_from_path) and not bool(cfg.training.resume)
        if resume_from_path:
            resume_from_path = pathlib.Path(os.path.expanduser(str(resume_from_path)))
            if not resume_from_path.is_file():
                raise FileNotFoundError(f"resume_from_path not found: {resume_from_path}")
            print(f"Resuming from checkpoint {resume_from_path}")
            if cfg.training.resume:
                # Same-run resume: restore the complete training state.
                self.load_checkpoint(path=resume_from_path)
            else:
                # Cross-stage transfer: keep this stage's output directory and
                # optimizer, while inheriting model weights and epoch. Loading
                # _output_dir/global_step would break the new stage's save/LR state.
                self.load_checkpoint(
                    path=resume_from_path,
                    exclude_keys=('optimizer',),
                    include_keys=('epoch',),
                )
                # The stage gate evaluates ema_model. Promote exactly those
                # validated weights into the train model for the next stage.
                if self.ema_model is not None:
                    self.model.load_state_dict(self.ema_model.state_dict())
                    print(
                        "[Transfer] initialized train model from validated "
                        "EMA checkpoint weights"
                    )
        elif cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        dataset: BaseLowdimDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseLowdimDataset)
        if hasattr(dataset, 'gripper_close_threshold'):
            _apply_gripper_close_semantics(
                self.model,
                dataset.gripper_close_threshold,
                dataset.gripper_close_is_greater,
            )
            if self.ema_model is not None:
                _apply_gripper_close_semantics(
                    self.ema_model,
                    dataset.gripper_close_threshold,
                    dataset.gripper_close_is_greater,
                )
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        scheduler_num_epochs = int(
            cfg.training.get('lr_scheduler_num_epochs', cfg.training.num_epochs)
        )
        if scheduler_num_epochs <= 0:
            raise ValueError(
                "training.lr_scheduler_num_epochs must be positive"
            )
        gradient_accumulate_every = int(
            cfg.training.gradient_accumulate_every
        )
        if gradient_accumulate_every <= 0:
            raise ValueError(
                "training.gradient_accumulate_every must be positive"
            )
        scheduler_total_steps = (
            len(train_dataloader) * scheduler_num_epochs
        ) // gradient_accumulate_every
        if scheduler_total_steps <= 0:
            raise ValueError(
                "LR scheduler has no training steps; check dataloader size "
                "and gradient accumulation"
            )

        scheduler_start_epoch = int(
            cfg.training.get('lr_scheduler_start_epoch', 0)
        )
        if scheduler_start_epoch < 0:
            raise ValueError(
                "training.lr_scheduler_start_epoch must be non-negative"
            )

        # global_step is a batch counter saved by the previous process. It
        # cannot be used as scheduler progress after a dataset/prefix change,
        # because the number of batches per epoch may be different. That made
        # a resumed ToolHang run at epoch 1100 look almost finished against a
        # 3001-epoch horizon and collapsed its LR to ~5e-8. A transferred
        # curriculum stage also inherits the source checkpoint's absolute
        # epoch, so its scheduler must count only epochs added by this stage.
        # Reconstruct progress from stage-relative epoch and the current loader.
        if cross_stage_transfer:
            scheduler_resume_step = 0
        else:
            completed_stage_epochs = max(
                0, int(self.epoch) - scheduler_start_epoch
            )
            completed_batches = (
                completed_stage_epochs * len(train_dataloader)
            )
            scheduler_resume_step = (
                completed_batches + gradient_accumulate_every - 1
            ) // gradient_accumulate_every
            scheduler_resume_step = min(
                scheduler_resume_step, scheduler_total_steps - 1
            )

        # Optimizer checkpoints can also carry an initial_lr from an obsolete
        # chunk scheduler. Anchor every resumed scheduler to this run's
        # configured base LR before LambdaLR reads the parameter groups.
        configured_base_lr = float(cfg.optimizer.lr)
        for param_group in self.optimizer.param_groups:
            param_group['initial_lr'] = configured_base_lr
            param_group['lr'] = configured_base_lr

        print(
            "[Training] fixed LR scheduler horizon: "
            f"{scheduler_num_epochs} epochs "
            f"(this process stops at {cfg.training.num_epochs}); "
            f"resume_step={scheduler_resume_step}/"
            f"{scheduler_total_steps}, epoch={self.epoch}, "
            f"stage_start_epoch={scheduler_start_epoch}, "
            f"saved_global_step={self.global_step}"
        )
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=scheduler_total_steps,
            last_epoch=scheduler_resume_step - 1,
        )
        effective_lr = float(lr_scheduler.get_last_lr()[0])
        scheduler_progress = (
            float(scheduler_resume_step) / float(scheduler_total_steps)
        )
        print(
            "[Training] LR scheduler initialized: "
            f"effective_lr={effective_lr:.8g}, "
            f"progress={scheduler_progress:.4f}"
        )
        if (
            scheduler_resume_step > int(cfg.training.lr_warmup_steps)
            and scheduler_progress < 0.95
            and effective_lr < configured_base_lr * 1e-4
        ):
            raise RuntimeError(
                "LR scheduler initialized at an effectively zero learning "
                f"rate ({effective_lr:.8g}) while only "
                f"{scheduler_progress:.1%} through this stage. Refusing to "
                "continue a silently stalled run."
            )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)
            ema.optimization_step = max(0, int(self.global_step))
            ema.decay = ema.get_decay(ema.optimization_step)
            print(
                "[Training] restored EMA schedule at "
                f"optimization_step={ema.optimization_step}, "
                f"decay={ema.decay:.6f}"
            )

        # configure env runner only when rollout is enabled. This allows
        # dataset-only training on systems without mujoco_py / robosuite.
        env_runner: BaseLowdimRunner = None
        rollout_enabled = cfg.training.rollout_every is not None
        if rollout_enabled:
            env_runner = hydra.utils.instantiate(
                cfg.task.env_runner, output_dir=self.output_dir
            )
            assert isinstance(env_runner, BaseLowdimRunner)

        # configure logging
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging,
        )
        wandb.config.update({"output_dir": self.output_dir})

        # ── 資料集分割統計 ────────────────────────────────────────
        n_train_samples = len(dataset)
        n_val_samples = len(val_dataset)
        print(f"[Dataset] train samples: {n_train_samples} | val samples: {n_val_samples}")
        wandb.config.update({
            "n_train_samples": n_train_samples,
            "n_val_samples": n_val_samples,
        }, allow_val_change=True)
        # 若 dataset 支援 split_info（SampConcatLowdimDataset），記錄逐子集統計
        if hasattr(dataset, 'split_info'):
            for info in dataset.split_info():
                print(
                    f"  [{info['class']}] idx={info['dataset_idx']} "
                    f"train={info['n_train_samples']} val={info['n_val_samples']}"
                )

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk,
        )

        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            if (
                cross_stage_transfer
                and rollout_enabled
                and bool(cfg.training.get('rollout_before_training', False))
            ):
                policy = self.ema_model if cfg.training.use_ema else self.model
                policy.eval()
                policy.reset()
                print(
                    f"[Transfer] running pre-training rollout at epoch={self.epoch}"
                )
                baseline_log = {
                    'epoch': self.epoch,
                    'global_step': self.global_step,
                    'transfer/pretrain_rollout': True,
                }
                try:
                    baseline_log.update(env_runner.run(policy))
                except Exception as exc:
                    if (
                        not _is_mujoco_instability_error(exc)
                        or not hasattr(env_runner, "recover_after_worker_error")
                    ):
                        raise
                    print(
                        "[WARN] Skipping the pre-training transfer rollout "
                        f"because MuJoCo became unstable: {exc}"
                    )
                    env_runner.recover_after_worker_error()
                    baseline_log[
                        "rollout/mujoco_instability_skipped"
                    ] = 1.0
                finally:
                    # runner may move policy to CPU before an exception.
                    policy.to(device)
                wandb_run.log(baseline_log, step=self.global_step)
                json_logger.log(baseline_log)

            if self.epoch >= cfg.training.num_epochs:
                print(
                    f"Checkpoint epoch {self.epoch} already reached "
                    f"target num_epochs={cfg.training.num_epochs}; nothing to train."
                )

            while self.epoch < cfg.training.num_epochs:
                freeze_until = cfg.training.get(
                    'freeze_encoder_until_epoch', None)
                encoder_trainable = (
                    freeze_until is None or self.epoch >= int(freeze_until)
                )
                changed = _set_samp_encoder_trainable(
                    self.model, encoder_trainable)
                if changed:
                    state = 'unfrozen' if encoder_trainable else 'frozen'
                    print(
                        f"[Transfer] SAMP encoder {state} at epoch={self.epoch} "
                        f"(freeze_until={freeze_until})"
                    )
                self.model.train()
                step_log = dict()

                # ========= train for this epoch ==========
                train_losses = list()
                with tqdm.tqdm(
                    train_dataloader,
                    desc=f"Training epoch {self.epoch}",
                    leave=False,
                    mininterval=cfg.training.tqdm_interval_sec,
                ) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        raw_loss = self.model.compute_loss(batch)
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        loss.backward()

                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()

                        if cfg.training.use_ema:
                            ema.step(self.model)

                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0],
                        }

                        is_last_batch = batch_idx == (len(train_dataloader) - 1)
                        if not is_last_batch:
                            wandb_run.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                        if (
                            cfg.training.max_train_steps is not None
                            and batch_idx >= cfg.training.max_train_steps - 1
                        ):
                            break

                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss

                # ========= eval for this epoch ==========
                policy = self.model
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # run rollout after the configured warmup epoch
                rollout_start_epoch = cfg.training.get('rollout_start_epoch', 0)
                if (
                    rollout_enabled
                    and self.epoch >= rollout_start_epoch
                    and self.epoch % cfg.training.rollout_every == 0
                ):
                    try:
                        runner_log = env_runner.run(policy)
                        step_log.update(runner_log)
                    except Exception as exc:
                        if (
                            not _is_mujoco_instability_error(exc)
                            or not hasattr(env_runner, "recover_after_worker_error")
                        ):
                            raise
                        print(
                            "[WARN] Skipping this rollout because MuJoCo became "
                            f"unstable: {exc}"
                        )
                        env_runner.recover_after_worker_error()
                        step_log["rollout/mujoco_instability_skipped"] = 1.0
                    finally:
                        # runner may move policy to CPU before an exception.
                        policy.to(device)

                # run validation
                if self.epoch % cfg.training.val_every == 0:
                    with torch.no_grad():
                        val_losses = list()
                        with tqdm.tqdm(
                            val_dataloader,
                            desc=f"Validation epoch {self.epoch}",
                            leave=False,
                            mininterval=cfg.training.tqdm_interval_sec,
                        ) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                loss = self.model.compute_loss(batch)
                                val_losses.append(loss)
                                if (
                                    cfg.training.max_val_steps is not None
                                    and batch_idx >= cfg.training.max_val_steps - 1
                                ):
                                    break
                        if val_losses:
                            val_loss = torch.mean(torch.tensor(val_losses)).item()
                            step_log['val_loss'] = val_loss

                # run sampling on a training batch
                if self.epoch % cfg.training.sample_every == 0:
                    with torch.no_grad():
                        batch = train_sampling_batch
                        obs_dict = {'obs': batch['obs']}
                        gt_action = batch['action']
                        # reset warm-start buffer so sampling is deterministic
                        policy.reset()
                        result = policy.predict_action(obs_dict)
                        if cfg.pred_action_steps_only:
                            pred_action = result['action']
                            start = cfg.n_obs_steps - 1
                            end = start + cfg.n_action_steps
                            gt_action = gt_action[:, start:end]
                        else:
                            pred_action = result['action_pred']
                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                        step_log['train_action_mse_error'] = mse.item()
                        del batch, obs_dict, gt_action, result, pred_action, mse

                # checkpoint
                if self.epoch % cfg.training.checkpoint_every == 0:
                    mean_test_score = step_log.get("test/mean_score", 0.0)
                    mean_train_score = step_log.get("train/mean_score", 0.0)
                    stage_grasp_rate = step_log.get("test/stage_grasp_rate", 0.0)
                    stage_insert_rate = step_log.get("test/stage_insert_rate", 0.0)
                    stage_full_rate = step_log.get("test/stage_full_rate", 0.0)
                    self.save_checkpoint(
                        f"{self.output_dir}/checkpoints/"
                        f"epoch={self.epoch}_test_score={mean_test_score:.3f}"
                        f"_train_score={mean_train_score:.3f}"
                        f"_grasp={stage_grasp_rate:.3f}"
                        f"_insert={stage_insert_rate:.3f}"
                        f"_full={stage_full_rate:.3f}.ckpt"
                    )
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    metric_dict = {k.replace('/', '_'): v for k, v in step_log.items()}
                    monitor_key = cfg.checkpoint.topk.monitor_key
                    if monitor_key not in metric_dict:
                        print(f"[WARN] monitor_key {monitor_key} not in metric_dict, skip topk checkpoint.")
                    else:
                        topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)
                        if topk_ckpt_path is not None:
                            self.save_checkpoint(path=topk_ckpt_path)

                # end of epoch bookkeeping
                wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1
