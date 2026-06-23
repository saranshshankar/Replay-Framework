"""CLI entry point for the replay platform."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import click

from replay.data_manager import resolve_local_bag, resolve_s3_bag
from replay.env_setup import DEFAULT_BUILD_JOBS, setup_environment
from replay.module_config import (
    load_checkout_paths,
    load_module_config,
    missing_preflight_assets,
)
from replay.runner import run_replay
from replay.version_manager import load_version_spec

DEFAULT_CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"
DEFAULT_VERSION_YAML_NAME = "default.yaml"

# Map the report generator's B9 exit code to a human-readable verdict for the
# CLI echo. 0 = PASS, 1 = quality FAIL, 2 = INVALID RUN (validity tier breach).
_VERDICT_LABELS = {0: "PASS", 1: "FAIL", 2: "INVALID RUN"}


def _build_metrics_cfg(module_spec) -> dict:
    """Build the cfg dict every metric plugin receives (WR-02 config threading).

    Carries not just the topic lists but the per-topic expected_hz map, the depth
    output topics, and the diagnostics topic — so faithfulness (01-12), DepthMetric
    (01-13) and LatencyMetric (01-13) stop falling back to broken defaults
    (flat-10Hz / all-output-topics / no-diagnostics). Kept as a tiny pure function
    so it is unit-testable without a bag.
    """
    return {
        "input_topics": module_spec.input_topics,
        "output_topics": module_spec.output_topics,
        "expected_hz": module_spec.expected_hz,        # per-topic map (01-12 reads it)
        "depth_topics": module_spec.depth_topics,      # 01-13 DepthMetric reads it
        "diagnostics_topic": module_spec.diagnostics_topic,  # 01-13 LatencyMetric reads it
        "gap_tolerance": module_spec.gap_tolerance,    # 01-20 faithfulness per-topic breach factor
        "latency_stage": module_spec.latency_stage,    # 01-21 LatencyMetric diagnostics op name
    }


def _resolve_baseline_bag(
    baseline: Optional[Path], module_spec, configs_dir: Path
) -> Optional[Path]:
    """Resolve a ``--baseline`` value to a readable baseline bag path.

    Two paths (MTRC-04, plan 01-15):
    - a local rosbag2 directory passed directly as ``--baseline <path>`` (the cheap
      CI/dev path — no S3, the curated golden is already on disk); used verbatim.
    - any other value is treated as the pinned-golden STRATEGY: resolve via
      ``BaselineManager(configs_dir).resolve(module)``, which reads the committed
      golden.yaml manifest and fetches the golden bag from S3.

    Returns None when ``baseline`` is None (no baseline requested).
    """
    if baseline is None:
        return None
    if Path(baseline).exists():
        return Path(baseline)
    from replay.metrics.baseline import BaselineManager

    ref = BaselineManager(configs_dir).resolve(module_spec.name)
    return ref.bag_path


def _run_metrics_pipeline(
    module_spec,
    bag_path: Path,
    output_dir: Path,
    baseline: Optional[Path] = None,
    configs_dir: Path = DEFAULT_CONFIGS_DIR,
    run_artifacts: Optional[dict] = None,
    run_viz: bool = False,
) -> int:
    """Run the registered perception plugins + faithfulness over an output bag,
    generate the report, and return the B9 exit code.

    Shared by `all --run-metrics` (post-replay) and the standalone `metrics`
    subcommand (offline, on a downloaded output bag — no replay/Docker/GPU).

    BASELINE (plan 01-15 / UAT gap 2): when ``baseline`` is given, every
    ``requires_baseline`` plugin is gated via ``compare(candidate, baseline, cfg)``
    instead of being skipped — this is the perception regression path
    (``mask_iou_vs_golden`` vs a pinned golden). FAIL-CLOSED: a baseline that yields
    no comparable frames produces a ``None`` headline scalar, so ``generator.py``
    emits a visible no-scalar row (never a silent pass / false green). When NO
    baseline is given, requires_baseline plugins are left OUT of the results so the
    generator emits its existing visible "skipped" row (warning-only until curated
    goldens land) — never a silent pass and never a false fail.
    """
    import replay.metrics.perception  # noqa: F401 — import triggers plugin self-registration (6 plugins)
    from replay.metrics.base import MetricResult
    from replay.metrics.bag_reader import BagReader
    from replay.metrics.registry import get_metric_plugins
    from replay.metrics.replay_faithfulness import ReplayFaithfulnessMetric
    from replay.metrics.report.generator import generate_report

    topics = list(module_spec.input_topics) + list(module_spec.output_topics)
    reader = BagReader(bag_path, topics)
    cfg = _build_metrics_cfg(module_spec)

    baseline_bag = _resolve_baseline_bag(baseline, module_spec, configs_dir)
    baseline_reader = BagReader(baseline_bag, topics) if baseline_bag is not None else None

    # Validity tier: faithfulness is invoked EXPLICITLY (it is deliberately not
    # registered as a quality plugin — 01-06), so it gates as the validity tier.
    faithfulness = ReplayFaithfulnessMetric().compute(reader, cfg)

    results = []
    for plugin_cls in get_metric_plugins(module_spec.name):
        plugin = plugin_cls()
        if getattr(plugin, "requires_baseline", False):
            if baseline_reader is None:
                # No baseline: leave the regression metric OUT of results so the
                # generator emits its visible "skipped" warning row (warning-only
                # until curated goldens land). Never a silent pass, never a false
                # fail. (UAT gap 2 fail-closed-on-absence.)
                continue
            # Regression path: gate via compare(candidate, baseline). The plugin's
            # top-level scalar (== plugin.name) is what generator.py enforces the
            # threshold against; a None scalar (zero comparable frames) fails closed.
            value = plugin.compare(reader, baseline_reader, cfg)
            results.append(
                MetricResult(
                    name=plugin.name,
                    module=module_spec.name,
                    value=value,
                    passed=False,
                    is_regression=True,
                )
            )
            continue
        value = plugin.compute(reader, cfg)
        results.append(
            MetricResult(
                name=plugin.name,
                module=module_spec.name,
                value=value,
                passed=True,
                is_regression=False,
            )
        )

    # Tier-3 viz (TIER3-VIZ-DESIGN §8): opt-in, CPU-only, rendered from the SAME
    # output bag + cfg, lazily (the import only happens here, never on the cheap
    # metrics path). Report-relative links (<output>/viz vs <output>/reports) are
    # threaded into the report so the Visualizations block points at the mp4s. Viz
    # NEVER affects the verdict/exit code (it is computed after the gate results).
    visualizations = None
    if run_viz:
        from replay.metrics.viz_runner import render_visualizations

        viz_paths = render_visualizations(module_spec.name, reader, cfg, output_dir)
        visualizations = [f"../viz/{Path(p).name}" for p in viz_paths] or None

    reports_dir = output_dir / "reports"
    # Debug-section pointer map (01-16's run_artifacts): the bag, the persisted
    # run logs (<output>/logs/, written by runner.py 01-17), and the report
    # itself. Default it from output_dir so the standalone `metrics` subcommand
    # also gets a logs pointer; all_cmd passes the real result.logs_dir.
    if run_artifacts is None:
        run_artifacts = {
            "bag": str(bag_path),
            "logs": str(output_dir / "logs"),
            "report": "report.html",
        }
    report_kwargs = dict(
        module=module_spec.name,
        run_id=str(output_dir.name),
        metric_results=results,
        output_dir=reports_dir,
        thresholds=module_spec.thresholds,
        faithfulness=faithfulness,
        run_artifacts=run_artifacts,
        visualizations=visualizations,
        # Tier-2 plots are DEFAULT on every run (the agreed scope — the report is
        # generated on every run anyway). Forwarding the reader is what triggers
        # generate_report's plot rendering; without it the report has no plots.
        reader=reader,
    )
    try:
        rc = generate_report(**report_kwargs)
    except TypeError:
        # Graceful degrade: an older generate_report that predates the additive
        # run_artifacts / visualizations params. Drop them and render without the
        # Debug pointer map / viz links rather than crash the run.
        report_kwargs.pop("run_artifacts", None)
        report_kwargs.pop("visualizations", None)
        rc = generate_report(**report_kwargs)
    click.echo(f"Metrics report: {reports_dir / 'report.html'}")
    click.echo(f"Verdict: {_VERDICT_LABELS[rc]}")
    return rc


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
    if result.logs_dir is not None:
        click.echo(f"Logs: {result.logs_dir}")


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
@click.option(
    "--baseline",
    type=click.Path(path_type=Path),
    default=None,
    help="Pinned-golden baseline for regression metrics (mask_iou_vs_golden). A "
    "local rosbag2 path is used directly; any other value resolves the golden via "
    "BaselineManager (configs/baselines/<module>/golden.yaml -> S3). Without it the "
    "regression metric is a visible warning-only 'skipped' row.",
)
@click.option("--run-viz", is_flag=True, default=False, help="(deferred — no-op in Phase 1)")
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
    baseline: Optional[Path],
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

    # B5 phase-0 fail-fast (contract C1): a missing TensorRT engine / camera
    # intrinsics LUT / config-current param tree must die HERE with the path
    # named — never as a deep on_activate mystery inside the container. This is
    # a named setup error (exit 3), distinct from the metrics verdict (1/2).
    missing = missing_preflight_assets(module_spec)
    if missing:
        click.echo("Pre-flight failed — missing assets:\n  " + "\n  ".join(missing), err=True)
        sys.exit(3)

    result = run_replay(module=module_spec, data=data_ref, output_dir=output_dir)
    click.echo(f"Output bag: {result.output_bag_path}")
    # The run logs (recorder.log + module.log) are persisted to <output>/logs/ by
    # the runner (01-17). Surface the path on success so the dev (and 01-18's CI
    # upload) can find them.
    if result.logs_dir is not None:
        click.echo(f"Logs: {result.logs_dir}")
    if result.exit_code != 0:
        # EXIT-CODE CONTRACT (SYSTEM-DESIGN-HLD-LLD B9): the process exit code is
        # the CI signal. 1 and 2 are RESERVED for the metrics verdict (quality
        # FAIL / INVALID RUN, plan 01-07). Any replay/setup failure maps to 3,
        # with the underlying container code echoed for diagnosis.
        click.echo(
            f"Replay failed (container exit code {result.exit_code}) "
            "— exiting 3 (setup/replay error)",
            err=True,
        )
        # B1 HEADLINE ASK (GAP c): the failure branch is EXACTLY where the logs
        # are the evidence — point the developer at them before exiting 3.
        if result.logs_dir is not None:
            click.echo(f"Replay logs: {result.logs_dir}", err=True)
        sys.exit(3)

    if run_metrics:
        # Thread the persisted logs into the report's Debug section (01-16
        # run_artifacts). Build it here from the real result.logs_dir so the
        # report's pointer map reflects this run's actual logs location.
        run_artifacts = {
            "bag": str(result.output_bag_path),
            "logs": str(result.logs_dir) if result.logs_dir is not None
            else str(output_dir / "logs"),
            "report": "report.html",
        }
        rc = _run_metrics_pipeline(
            module_spec, result.output_bag_path, output_dir,
            baseline=baseline, configs_dir=configs_dir,
            run_artifacts=run_artifacts, run_viz=run_viz,
        )
        if rc != 0:
            sys.exit(rc)  # 1 = quality FAIL, 2 = INVALID RUN — B9 contract
    elif run_viz:
        # --run-viz WITHOUT --run-metrics: render the Tier-3 videos standalone
        # from the output bag (no report/gate). Viz is CPU-only and never affects
        # any exit code (TIER3-VIZ-DESIGN §9).
        _run_viz_pipeline(module_spec, result.output_bag_path, output_dir)


@main.command("metrics")
@click.option("--module", required=True, type=click.Choice(["perception", "navigation", "manipulation"]))
@click.option("--bag", "bag_path", required=True, type=click.Path(path_type=Path, exists=True))
@click.option("--output", "output_dir", required=True, type=click.Path(path_type=Path))
@click.option("--configs-dir", type=click.Path(path_type=Path, exists=True), default=DEFAULT_CONFIGS_DIR)
@click.option(
    "--baseline",
    type=click.Path(path_type=Path),
    default=None,
    help="Pinned-golden baseline for regression metrics (mask_iou_vs_golden). A "
    "local rosbag2 path is used directly; any other value resolves the golden via "
    "BaselineManager (configs/baselines/<module>/golden.yaml -> S3). Without it the "
    "regression metric is a visible warning-only 'skipped' row.",
)
@click.option("--run-viz", is_flag=True, default=False,
              help="Also render the Tier-3 debug videos ([viz] extra) and link them in the report.")
def metrics_cmd(
    module: str, bag_path: Path, output_dir: Path, configs_dir: Path,
    baseline: Optional[Path], run_viz: bool,
) -> None:
    """Offline evaluation of an EXISTING output bag: plugins -> criteria -> report + exit code.

    The CHEAP runner (B9 CLI table / CI-01): no replay, no Docker, no GPU, and
    no preflight asset gate (it needs no robot assets) — just the offline
    metrics pipeline over a downloaded output bag, exiting per the B9 verdict.

    Pass ``--baseline <golden-bag>`` to run the regression metric
    (``mask_iou_vs_golden``) via compare(); without it that metric is a visible
    warning-only "skipped" row (UAT gap 2 / plan 01-15).

    Pass ``--run-viz`` to ALSO render the Tier-3 debug videos (needs the ``[viz]``
    extra). Viz is opt-in, CPU-only, and never affects the verdict/exit code
    (TIER3-VIZ-DESIGN §9).
    """
    module_spec = load_module_config(module, configs_dir / "modules")
    output_dir.mkdir(parents=True, exist_ok=True)
    rc = _run_metrics_pipeline(
        module_spec, bag_path, output_dir, baseline=baseline, configs_dir=configs_dir,
        run_viz=run_viz,
    )
    if rc != 0:
        sys.exit(rc)  # 1 = quality FAIL, 2 = INVALID RUN — B9 contract


def _run_viz_pipeline(module_spec, bag_path: Path, output_dir: Path) -> list:
    """Render a module's Tier-3 viz from an EXISTING output bag (no replay/gate).

    Shared by the standalone ``viz`` subcommand and ``all --run-viz`` (without
    metrics). Builds the same read-once ``BagReader`` + metrics cfg the plugins
    expect, then defers to the lazy ``render_visualizations`` runner.
    """
    from replay.metrics.bag_reader import BagReader
    from replay.metrics.viz_runner import render_visualizations

    topics = list(module_spec.input_topics) + list(module_spec.output_topics)
    reader = BagReader(bag_path, topics)
    cfg = _build_metrics_cfg(module_spec)
    paths = render_visualizations(module_spec.name, reader, cfg, output_dir)
    if paths:
        click.echo(f"Generated {len(paths)} visualization(s) in {Path(output_dir) / 'viz'}:")
        for p in paths:
            click.echo(f"  {p}")
    else:
        click.echo("No visualizations generated (see message above).")
    return paths


@main.command("viz")
@click.option("--module", required=True, type=click.Choice(["perception", "navigation", "manipulation"]))
@click.option("--bag", "bag_path", required=True, type=click.Path(path_type=Path, exists=True))
@click.option("--output", "output_dir", required=True, type=click.Path(path_type=Path))
@click.option("--configs-dir", type=click.Path(path_type=Path, exists=True), default=DEFAULT_CONFIGS_DIR)
def viz_cmd(module: str, bag_path: Path, output_dir: Path, configs_dir: Path) -> None:
    """Render Tier-3 debug videos offline from an EXISTING output bag.

    CPU-only, no replay/Docker/GPU/gate (TIER3-VIZ-DESIGN §2). Writes
    ``<output>/viz/*.mp4`` (overlap_video + the per-camera combined grid). Needs the
    ``[viz]`` extra (``pip install module_replay[viz]``); a missing encoder prints an
    install hint and exits cleanly without affecting anything else.
    """
    module_spec = load_module_config(module, configs_dir / "modules")
    output_dir.mkdir(parents=True, exist_ok=True)
    _run_viz_pipeline(module_spec, bag_path, output_dir)


if __name__ == "__main__":
    main()
