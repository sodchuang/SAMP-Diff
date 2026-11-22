# Experimental fixed semantic profiles

Scope: one demonstration contains several sequential subtasks, not mixing
independent Lift/Can datasets. A single learned generator still coordinates all
action dimensions. The new option changes its source prior, not inference steps.
All features are opt-in; old configs and model weights keep their default path.

## Contracts

- Stored absolute HDF5 actions remain 7D per arm (xyz, axis-angle, gripper).
  The existing dataset rotation transformer produces 10D per arm for the policy.
- Only explicit absolute 10D/20D policy layouts are supported by fixed profiles.
- Shared profiles: grasp / transport / align / release. Constants are starting
  hypotheses, **not** parameters shown to work or optimized for a benchmark.
- Each arm chooses its own profile. No action tensor is cached by phase alone.
  Runtime cache ownership is stream + task + episode + contiguous step.
- Phase changes selectively discard groups; align discards incoming rotation
  history, grasp/release use fresh gripper source. All groups remain jointly
  generated. No gripper latch or hard-coded completion controller is added.
- Training uses aligned teacher-forced history, not a persistent shuffled-batch
  cache. This is a training/inference approximation, not on-policy history.
- History modes `none`, `inherit`, `reset`, `selective` provide controlled
  comparisons with the same generator, step count and fixed profile bank.

## Prepare

From `SAMP_Diff_v1`, with the existing simulation environment installed:

```bash
python scripts/convert_robomimic_delta_to_abs.py \
  -i data/mimicgen/core/stack_d1.hdf5 \
  -o data/mimicgen/core/stack_d1_abs.hdf5 --verify-demos 3

python scripts/prepare_action_slices.py \
  --dataset data/mimicgen/core/stack_d1_abs.hdf5 \
  --output data/mimicgen/core/stack_d1_phases.json --weak-event-labels
```

Already converted files do not need conversion again. Conversion restores every
recorded simulator state and reads controller goals, never a cumulative sum.
Verification reports one-step EEF errors (currently robot0), not task success.

The event labeler is a **weak offline heuristic**: closure -> transport, a fixed
pre-release window -> align, opening -> release. It uses future actions for
offline annotation only. It is not evidence of object alignment or task success.
For paper claims prefer manually verified boundaries. The JSON can be manually
authored using exact contiguous `[start,end)` segments and those four names:

```json
{"schema_version":1,"phase_source":"manual","task_id":0,"action_dim":10,
 "demos":{"demo_0":{"length":100,"arms":{"arm0":[
 {"start":0,"end":30,"phase":"grasp"},
 {"start":30,"end":65,"phase":"transport"},
 {"start":65,"end":90,"phase":"align"},
 {"start":90,"end":100,"phase":"release"}]}}}}
```

Include **every** demonstration. For 20D include arm0 and arm1 separately.
Prefix truncation is rejected; episode lengths must match the sidecar.

## Train

```bash
python scripts/train_fixed_phase_task.py --config can_ph \
  --dataset data/mimicgen/core/stack_d1_abs.hdf5 \
  --manifest data/mimicgen/core/stack_d1_phases.json \
  --output data/outputs/fixed_phase_stack_seed42 \
  --epochs 5000 --seed 42 --device cuda:0
```

Use `--config transport_ph` and a converted dual-arm dataset/manifest for 20D.
Observation dimensions are read using that config's observation keys. No joint
task completion, improvement, or single/dual-arm simulator validation is claimed.
Use a new output directory. `--dry-run` writes the composed config only.
This launcher intentionally disables simulator rollouts; diagnostics on labeled
training batches are oracle-phase diagnostics, **not online success rates**.
Run it inside your allocated Slurm job as with existing Python training commands.

For the existing h200q cluster, submit from `SAMP_Diff_v1` (no interactive srun):

```bash
sbatch --account=ChingWenMa scripts/slurm_fixed_phase.sh \
  --config can_ph --dataset data/mimicgen/core/stack_d1_abs.hdf5 \
  --manifest data/mimicgen/core/stack_d1_phases.json \
  --output data/outputs/fixed_phase_stack_seed42 --epochs 5000
```

Set `SAMP_PYTHON` before submitting if the Conda environment is in another path.
The example account/partition must match your own allocation. This launcher
starts a fresh run; it does not automatically resubmit after the Slurm time limit.

## Online inference integration

Existing `eval.py` alone is NOT sufficient for this experimental option.
The caller must provide `obs` plus `phase_id` shaped `(B,n_arms)` and one integer
per row for `stream_id`, `task_id`, `episode_id`, `step`. Phase IDs are 0..3 in
the order above. Choose phases using current observations, explicit manual input
or an independently evaluated causal detector. Never replay future demo labels
as if they were a deployed detector. Missing context fails loudly.

Advance step by the actual configured action chunk length. A jump resets reuse.
Call `policy.reset()` between full vectorized rollouts; individual finished
streams can use `policy.fixed_phase_prior.clear([stream_id])`. Reordered batch
rows retain their own history via stream IDs. The four profiles and boundary
rules are in `diffusion_policy/common/fixed_phase_prior.py`; profiles may also
be supplied in `policy.fixed_phase_prior_params.profiles` (not learned).

## CPU checks

```bash
python -m unittest discover -s tests -p test_fixed_phase_prior.py -v
```

Tests cover isolation, reordering, arm-local transitions, spectral source
construction and slice coverage. They do not replace MuJoCo task evaluation.
