#!/usr/bin/env python3
"""Create an absolute-action copy of a RoboMimic / MimicGen HDF5 dataset."""

import copy
import json
import pathlib
import shutil
import sys

import click
import h5py
from tqdm import tqdm


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diffusion_policy.common.robomimic_util import (  # noqa: E402
    RobomimicAbsoluteActionConverter,
)


def _mark_dataset_absolute(output_file):
    """Update RoboMimic metadata so generic playback also uses abs control."""
    data_group = output_file["data"]
    raw_env_args = data_group.attrs.get("env_args")
    if raw_env_args is None:
        return False
    if isinstance(raw_env_args, bytes):
        raw_env_args = raw_env_args.decode("utf-8")
    env_args = copy.deepcopy(json.loads(raw_env_args))
    controller = env_args.setdefault("env_kwargs", {}).get("controller_configs")
    if isinstance(controller, dict):
        controller["control_delta"] = False
    else:
        return False
    data_group.attrs["env_args"] = json.dumps(env_args)
    return True


@click.command()
@click.option("-i", "--input", "input_path", required=True, type=click.Path(path_type=pathlib.Path))
@click.option("-o", "--output", "output_path", required=True, type=click.Path(path_type=pathlib.Path))
@click.option("--overwrite", is_flag=True, help="Replace an existing output file.")
@click.option(
    "--verify-demos",
    default=3,
    show_default=True,
    type=click.IntRange(min=0),
    help="Replay this many converted demos through the absolute controller.",
)
def main(input_path, output_path, overwrite, verify_demos):
    input_path = input_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if not input_path.is_file():
        raise click.ClickException(f"input dataset not found: {input_path}")
    if input_path == output_path:
        raise click.ClickException("input and output must be different files")
    if output_path.exists() and not overwrite:
        raise click.ClickException(
            f"output already exists: {output_path}; use --overwrite to replace it"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    converter = RobomimicAbsoluteActionConverter(
        input_path, create_abs_env=verify_demos > 0
    )
    try:
        demo_keys = list(converter.demo_keys)
        converted = {}
        verification = {}
        for index, demo_key in enumerate(tqdm(demo_keys, desc="Converting delta -> abs")):
            if index < verify_demos:
                actions, metrics = converter.convert_and_evaluate_key(demo_key)
                verification[demo_key] = metrics
            else:
                actions = converter.convert_key(demo_key)
            converted[demo_key] = actions
    finally:
        converter.close()

    temporary_path = output_path.with_suffix(output_path.suffix + ".partial")
    if temporary_path.exists():
        temporary_path.unlink()
    shutil.copy2(input_path, temporary_path)
    try:
        with h5py.File(temporary_path, "r+") as output_file:
            for demo_key, actions in tqdm(converted.items(), desc="Writing actions"):
                output_file[f"data/{demo_key}/actions"][:] = actions
            output_file.attrs["action_representation"] = "absolute_eef_pose_axis_angle"
            output_file.attrs["converted_from"] = str(input_path)
            metadata_updated = _mark_dataset_absolute(output_file)

        temporary_path.replace(output_path)
    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()
        raise

    report_path = output_path.with_suffix(output_path.suffix + ".conversion.json")
    report_path.write_text(
        json.dumps(
            {
                "input": str(input_path),
                "output": str(output_path),
                "demos": len(demo_keys),
                "env_metadata_control_delta_updated": metadata_updated,
                "verification": verification,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    click.echo(f"[DONE] absolute-action dataset: {output_path}")
    click.echo(f"[DONE] conversion report: {report_path}")


if __name__ == "__main__":
    main()
