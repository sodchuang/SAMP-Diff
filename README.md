# SAMP-Diff

Spectral-Adaptive Modulated Prior Diffusion experiments for low-dimensional robot manipulation tasks.

## VM Quick Start

```bash
git clone https://github.com/sodchuang/SAMP-Diff.git
cd SAMP-Diff/SAMP_Diff_v1

conda env create -f environment_h100_py310.yml
conda activate robodiff310

# Install CUDA PyTorch first for the target VM, then:
pip install -r requirements_runtime_h100.txt
pip install -r requirements_train_h100.txt
pip install -e .
```

Do not use `--no-deps` for the base runtime or training requirements. Use dependency isolation only for the legacy MuJoCo / robosuite stack.

For robomimic / MimicGen rollout evaluation, install simulator extras after the base training import works:

```bash
bash scripts/install_mujoco_robosuite_stack.sh
```

## Data Layout

The launchers assume datasets are stored under `SAMP_Diff_v1/data/`, which is ignored by git.

For MimicGen single-task runs:

```text
SAMP_Diff_v1/data/mimicgen/core/stack_d1.hdf5
SAMP_Diff_v1/data/mimicgen/core/nut_assembly_d0.hdf5
SAMP_Diff_v1/data/mimicgen/core/threading_d2.hdf5
```

## MimicGen Single-Task Runs

Check environment registration and dataset shapes before allocating long GPU jobs:

```bash
cd SAMP_Diff_v1
bash scripts/run_mimicgen_single_tasks.sh --check
```

Run the default three single tasks:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_mimicgen_single_tasks.sh
```

Useful overrides:

```bash
TASKS="stack_d1" NUM_EPOCHS=4000 N_ENVS=28 N_TEST=50 \
CUDA_VISIBLE_DEVICES=0 bash scripts/run_mimicgen_single_tasks.sh
```

Outputs default to:

```text
SAMP_Diff_v1/data/outputs/mimicgen_single_tasks/
```

## Experiment Launchers

Common scripts:

```bash
bash scripts/run_robomimic_benchmarks.sh
bash scripts/run_transport_final_then_generalization.sh
bash scripts/run_dataset_source_generalization.sh
bash scripts/run_square_semantic_ablation.sh
```

Most scripts expose task, seed, output, and rollout settings through environment variables at the top of the file.

## Notes

- Checkpoints, datasets, logs, videos, and analysis outputs are intentionally ignored by git.
- Use `WANDB_MODE=online` only when the VM has a configured Weights & Biases login.
- For quick smoke tests, reduce `NUM_EPOCHS`, `N_ENVS`, and `N_TEST` before starting a full H100 run.
