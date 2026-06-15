"""CLI entry point for the replay platform."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import click

from replay.data_manager import resolve_local_bag, resolve_s3_bag
from replay.env_setup import DEFAULT_BUILD_JOBS, setup_environment
from replay.module_config import load_checkout_paths, load_module_config
from replay.runner import run_replay
from replay.version_manager import load_version_spec

DEFAULT_CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"
DEFAULT_VERSION_YAML_NAME = "default.yaml"


def _resolve_version_yaml(configs_dir: Path, version_yaml: Optional[Path]) -> Optional[Path]:
    """If --version-yaml isn't passed, use configs/versions/default.yaml when it exists."""
    if version_yaml is not None:
        return version_yaml
    fallback = configs_dir / "versions" / DEFAULT_VERSION_YAML_NAME
    return fallback if fallback.exists() else None


def _load_specs(module: str, configs_dir: Path, version_yaml: Optional[Path]):
    module_spec = load_module_config(module, configs_dir / "modules")
    version_spec = load_version_spec(_resolve_version_yaml(configs_dir, version_yaml))
    return version_spec, module_spec


@click.group()
def main() -> None:
    """Local module-wise replay platform for 10xCode."""


@main.command()
@click.option("--module", required=True, type=click.Choice(["perception", "navigation", "manipulation"]))
@click.option("--version-yaml", type=click.Path(path_type=Path, exists=True), default=None)
@click.option("--configs-dir", type=click.Path(path_type=Path, exists=True), default=DEFAULT_CONFIGS_DIR)
def validate(module: str, version_yaml: Optional[Path], configs_dir: Path) -> None:
    """Resolve and print the module + version specs without side-effects."""
    version_spec, module_spec = _load_specs(module, configs_dir, version_yaml)
    click.echo(f"Module: {module_spec.name}")
    click.echo(f"  Container: {module_spec.container}")
    click.echo(f"  Colcon package: {module_spec.colcon_package}")
    click.echo(f"  Input topics: {module_spec.input_topics}")
    click.echo(f"  Output topics: {module_spec.output_topics}")
    click.echo(f"10xCode branch: {version_spec.tenxcode_branch}")
    if version_spec.submodule_overrides:
        click.echo("Submodule overrides:")
        for o in version_spec.submodule_overrides:
            click.echo(f"  - {o.name}: {o.branch}")
    else:
        click.echo("Submodule overrides: (none - pinned from 10xCode branch)")


@main.command("fetch-data")
@click.option("--task-id")
@click.option("--robot-id")
@click.option("--local-bag", type=click.Path(path_type=Path, exists=True))
@click.option("--output", "output_dir", required=True, type=click.Path(path_type=Path))
@click.option("--s3-bucket", envvar="REPLAY_S3_BUCKET")
def fetch_data(
    task_id: Optional[str],
    robot_id: Optional[str],
    local_bag: Optional[Path],
    output_dir: Path,
    s3_bucket: Optional[str],
) -> None:
    """Resolve a rosbag: either from S3 (--task-id + --robot-id) or local (--local-bag)."""
    if local_bag is not None:
        ref = resolve_local_bag(local_bag)
    elif task_id and robot_id:
        if not s3_bucket:
            raise click.UsageError("REPLAY_S3_BUCKET env var or --s3-bucket flag required for S3 fetch")
        output_dir.mkdir(parents=True, exist_ok=True)
        ref = resolve_s3_bag(
            task_id=task_id,
            robot_id=robot_id,
            bucket=s3_bucket,
            dest_dir=output_dir,
        )
    else:
        raise click.UsageError(
            "Provide either --local-bag OR (--task-id AND --robot-id)"
        )
    click.echo(str(ref.local_path))


@main.command("setup-env")
@click.option("--module", required=True, type=click.Choice(["perception", "navigation", "manipulation"]))
@click.option("--version-yaml", type=click.Path(path_type=Path, exists=True), default=None)
@click.option("--configs-dir", type=click.Path(path_type=Path, exists=True), default=DEFAULT_CONFIGS_DIR)
@click.option("--partial-checkout", is_flag=True, default=False, help="Check out only the module's paths from configs/modules/checkout_paths.yaml instead of the full 10xCode tree.")
@click.option(
    "--build-jobs",
    type=click.IntRange(min=1),
    default=DEFAULT_BUILD_JOBS,
    show_default=True,
    help="Cap colcon parallel-workers and MAKEFLAGS -j to N processes. Default 2 - drop to 1 if the laptop still stalls.",
)
def setup_env(
    module: str,
    version_yaml: Optional[Path],
    configs_dir: Path,
    partial_checkout: bool,
    build_jobs: int,
) -> None:
    """Bring up the container, check out 10xCode + submodules, build the module."""
    version_spec, module_spec = _load_specs(module, configs_dir, version_yaml)
    paths_to_checkout = (
        load_checkout_paths(module, configs_dir / "modules") if partial_checkout else None
    )
    setup_environment(
        version_spec,
        module_spec,
        checkout_paths=paths_to_checkout,
        build_jobs=build_jobs,
    )
    click.echo(f"Environment ready for module={module}, branch={version_spec.tenxcode_branch}")


@main.command()
@click.option("--module", required=True, type=click.Choice(["perception", "navigation", "manipulation"]))
@click.option("--bag", "bag_path", required=True, type=click.Path(path_type=Path, exists=True))
@click.option("--output", "output_dir", required=True, type=click.Path(path_type=Path))
@click.option("--configs-dir", type=click.Path(path_type=Path, exists=True), default=DEFAULT_CONFIGS_DIR)
def run(module: str, bag_path: Path, output_dir: Path, configs_dir: Path) -> None:
    """Play the filtered bag through the module and record its outputs."""
    from replay.data_manager import DataRef

    module_spec = load_module_config(module, configs_dir / "modules")
    data_ref = DataRef(local_path=bag_path, source="local")
    result = run_replay(module=module_spec, data=data_ref, output_dir=output_dir)
    click.echo(f"Output bag: {result.output_bag_path}")
    click.echo(f"Exit code: {result.exit_code}")


@main.command("all")
@click.option("--module", required=True, type=click.Choice(["perception", "navigation", "manipulation"]))
@click.option("--task-id")
@click.option("--robot-id")
@click.option("--local-bag", type=click.Path(path_type=Path, exists=True))
@click.option("--version-yaml", type=click.Path(path_type=Path, exists=True), default=None)
@click.option("--output", "output_dir", required=True, type=click.Path(path_type=Path))
@click.option("--configs-dir", type=click.Path(path_type=Path, exists=True), default=DEFAULT_CONFIGS_DIR)
@click.option("--s3-bucket", envvar="REPLAY_S3_BUCKET")
@click.option("--partial-checkout", is_flag=True, default=False, help="Check out only the module's paths from configs/modules/checkout_paths.yaml.")
@click.option(
    "--build-jobs",
    type=click.IntRange(min=1),
    default=DEFAULT_BUILD_JOBS,
    show_default=True,
    help="Cap colcon parallel-workers and MAKEFLAGS -j to N processes. Default 2 - drop to 1 if the laptop still stalls.",
)
@click.option("--run-metrics", is_flag=True, default=False)
@click.option("--run-viz", is_flag=True, default=False)
def all_cmd(
    module: str,
    task_id: Optional[str],
    robot_id: Optional[str],
    local_bag: Optional[Path],
    version_yaml: Optional[Path],
    output_dir: Path,
    configs_dir: Path,
    s3_bucket: Optional[str],
    partial_checkout: bool,
    build_jobs: int,
    run_metrics: bool,
    run_viz: bool,
) -> None:
    """Run all pipeline stages end-to-end."""

    version_spec, module_spec = _load_specs(module, configs_dir, version_yaml)

    if local_bag is not None:
        data_ref = resolve_local_bag(local_bag)
    elif task_id and robot_id:
        if not s3_bucket:
            raise click.UsageError("REPLAY_S3_BUCKET env var or --s3-bucket flag required for S3 fetch")
        output_dir.mkdir(parents=True, exist_ok=True)
        data_ref = resolve_s3_bag(
            task_id=task_id, robot_id=robot_id, bucket=s3_bucket, dest_dir=output_dir,
        )
    else:
        raise click.UsageError("Provide --local-bag OR (--task-id AND --robot-id)")

    paths_to_checkout = (
        load_checkout_paths(module, configs_dir / "modules") if partial_checkout else None
    )
    setup_environment(
        version_spec,
        module_spec,
        checkout_paths=paths_to_checkout,
        build_jobs=build_jobs,
    )
    result = run_replay(module=module_spec, data=data_ref, output_dir=output_dir)
    click.echo(f"Output bag: {result.output_bag_path}")

    if run_metrics:
        click.echo("Metrics requested - not implemented yet.")
    if run_viz:
        click.echo("Viz requested - not implemented yet.")


if __name__ == "__main__":
    main()
