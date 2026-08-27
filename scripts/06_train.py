"""[必选] 按配置顺序训练多个种子并执行候选预检。"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
import re
import shlex
import signal
import subprocess
import sys
from pathlib import Path

from car_rl_train.workflow import (
    ROOT,
    atomic_write_json,
    completed_run_checkpoint,
    eligible_run_checkpoints,
    load_run_status,
    load_workflow_config,
    next_training_segment_iterations,
    project_relative_path,
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", type=Path, default=ROOT / "configs" / "train.yaml")
parser.add_argument("--seeds", nargs="+", type=int)
parser.add_argument("--single-seed", type=int, help="只执行一次训练，不创建 campaign")
parser.add_argument("--checkpoint", help="单次训练使用的恢复点")
parser.add_argument("--run-name", help="单次训练的运行名称")
parser.add_argument("--num_envs", type=int)
parser.add_argument("--max_iterations", type=int)
parser.add_argument("--extension_iterations", type=int)
parser.add_argument("--max_total_iterations", type=int)
parser.add_argument("--no_auto_extend", action="store_true")
parser.add_argument("--campaign-name")
args, extra = parser.parse_known_args()
workflow = load_workflow_config(args.config)

worker_module = "car_rl_train.train_worker"
seeds = args.seeds or workflow.get("seeds", [42, 43, 44])
initial_iterations = int(args.max_iterations or workflow.get("max_iterations", 8000))
extension_iterations = int(
    args.extension_iterations or workflow.get("extension_iterations", 2000)
)
max_total_iterations = int(
    args.max_total_iterations or workflow.get("max_total_iterations", 16000)
)
auto_extend = bool(workflow.get("auto_extend", True)) and not args.no_auto_extend
if initial_iterations <= 0 or extension_iterations <= 0:
    raise ValueError("initial and extension iterations must be positive")
if max_total_iterations < initial_iterations:
    raise ValueError("max_total_iterations must be >= initial max_iterations")
if args.single_seed is not None:
    command = [
        sys.executable,
        "-m",
        worker_module,
        "--config",
        str(args.config),
        "--seed",
        str(args.single_seed),
        "--max_iterations",
        str(initial_iterations),
    ]
    if args.num_envs is not None:
        command.extend(("--num_envs", str(args.num_envs)))
    if args.checkpoint:
        command.extend(("--checkpoint", args.checkpoint))
    if args.run_name:
        command.extend(("--run_name", args.run_name))
    command.extend(extra)
    print("running:", shlex.join(command), flush=True)
    raise SystemExit(subprocess.run(command, check=False, cwd=ROOT).returncode)
campaign_name = args.campaign_name or datetime.now().strftime("campaign_%Y%m%d_%H%M%S")
if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", campaign_name):
    raise ValueError("campaign-name contains unsafe characters")
experiment_root = ROOT / "logs" / "rsl_rl" / "car_local_navigation"
campaign_root = experiment_root / "campaigns"
campaign_root.mkdir(parents=True, exist_ok=True)
campaign_path = campaign_root / f"{campaign_name}.json"
if campaign_path.exists():
    raise FileExistsError(f"campaign already exists: {campaign_path}")
campaign = {
    "schema_version": 2,
    "name": campaign_name,
    "status": "running",
    "created_at": datetime.now().astimezone().isoformat(),
    "finished_at": None,
    "seeds": [int(seed) for seed in seeds],
    "training_budget": {
        "initial_iterations": initial_iterations,
        "extension_iterations": extension_iterations,
        "max_total_iterations": max_total_iterations,
        "auto_extend": auto_extend,
    },
    "runs": [],
    "checkpoints": [],
    "failure": None,
}
atomic_write_json(campaign_path, campaign)


def passes_screen(metrics: dict, validation: dict) -> bool:
    """检查 campaign 的小样本筛选线，不替代最终生产门禁。"""

    return (
        float(metrics.get("success_rate", -1.0))
        >= float(validation["success_rate_min"])
        and float(metrics.get("collision_rate", 1.0))
        <= float(validation["collision_rate_max"])
    )


def validate_seed_candidates(seed: int, checkpoints: list[Path]) -> dict:
    """在 core/dynamic 固定验证集上选择同一个双门禁 checkpoint。"""

    evaluate_script = ROOT / "scripts" / "07_evaluate.py"
    evaluate_config_path = ROOT / "configs" / "evaluate.yaml"
    evaluate_config = load_workflow_config(evaluate_config_path)
    validation_root = ROOT / "evaluations" / "campaign_validation" / campaign_name / f"seed{seed}"
    rankings = {}
    for suite_name in ("core", "dynamic"):
        suite = evaluate_config["suites"][suite_name]
        validation = suite["validation"]
        command = [
            sys.executable,
            str(evaluate_script),
            "--config",
            str(evaluate_config_path),
            "--suite",
            suite_name,
            "--episodes",
            str(validation["episodes"]),
            "--seed-start",
            str(validation["seed_range"][0]),
            "--seed-end",
            str(validation["seed_range"][1]),
            "--output-root",
            str(validation_root / suite_name),
            "--headless",
        ]
        for checkpoint in checkpoints:
            command.extend(("--checkpoint", str(checkpoint)))
        print("validating:", shlex.join(command), flush=True)
        subprocess.run(command, check=True, cwd=ROOT)
        ranking_path = validation_root / suite_name / "candidate_ranking.json"
        rankings[suite_name] = json.loads(ranking_path.read_text(encoding="utf-8"))

    by_suite = {
        name: {item["checkpoint"]: item for item in ranking}
        for name, ranking in rankings.items()
    }
    common = set(by_suite["core"]) & set(by_suite["dynamic"])
    qualified = []
    for checkpoint in common:
        core = by_suite["core"][checkpoint]["metrics"]
        dynamic = by_suite["dynamic"][checkpoint]["metrics"]
        if passes_screen(core, evaluate_config["suites"]["core"]["validation"]) and passes_screen(
            dynamic, evaluate_config["suites"]["dynamic"]["validation"]
        ):
            qualified.append((checkpoint, core, dynamic))
    if not qualified:
        diagnostics = {
            name: [
                {
                    "checkpoint": item["checkpoint"],
                    "success_rate": item["metrics"].get("success_rate"),
                    "collision_rate": item["metrics"].get("collision_rate"),
                }
                for item in ranking
            ]
            for name, ranking in rankings.items()
        }
        raise RuntimeError(
            f"seed={seed} has no checkpoint passing both validation screens: {diagnostics}"
        )
    qualified.sort(
        key=lambda item: (
            float(item[1]["collision_rate"]),
            float(item[2]["collision_rate"]),
            -0.5 * (float(item[1]["success_rate"]) + float(item[2]["success_rate"])),
        )
    )
    checkpoint, core, dynamic = qualified[0]
    return {
        "selected_checkpoint": checkpoint,
        "validation_root": project_relative_path(validation_root),
        "core_metrics": core,
        "dynamic_metrics": dynamic,
    }


def _experiment_directories() -> set[Path]:
    if not experiment_root.is_dir():
        return set()
    return {
        path.resolve()
        for path in experiment_root.iterdir()
        if path.is_dir() and path.name != "campaigns"
    }


def run_training_segment(
    seed: int,
    segment_index: int,
    iterations: int,
    resume_checkpoint: Path | None,
) -> tuple[Path, Path, dict]:
    """运行一个可审计训练段，并严格解析它唯一产生的运行目录。"""

    before = _experiment_directories()
    suffix = f"{campaign_name}_seed{seed}"
    if segment_index:
        suffix += f"_part{segment_index + 1}"
    command = [
        sys.executable,
        "-m",
        worker_module,
        "--config",
        str(args.config),
        "--seed",
        str(seed),
        "--run_name",
        suffix,
        "--max_iterations",
        str(iterations),
        "--headless",
    ]
    if resume_checkpoint is not None:
        command.extend(("--checkpoint", str(resume_checkpoint)))
    if args.num_envs is not None:
        command.extend(("--num_envs", str(args.num_envs)))
    command.extend(extra)
    print("running:", shlex.join(command), flush=True)
    process = subprocess.Popen(command, cwd=ROOT, start_new_session=True)
    try:
        return_code = process.wait()
    except KeyboardInterrupt:
        os.killpg(process.pid, signal.SIGINT)
        process.wait()
        raise
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)
    after = _experiment_directories()
    new_runs = [
        path
        for path in sorted(after - before)
        if (load_run_status(path) or {}).get("seed") == int(seed)
    ]
    if len(new_runs) != 1:
        raise RuntimeError(
            f"expected exactly one new completed run for seed={seed}, got {new_runs}"
        )
    run_dir = new_runs[0]
    final_checkpoint = completed_run_checkpoint(run_dir)
    if final_checkpoint is None:
        raise RuntimeError(f"training segment is not completed: {run_dir}")
    return run_dir, final_checkpoint, load_run_status(run_dir) or {}

try:
    for seed in seeds:
        seed_record = {
            "seed": int(seed),
            "status": "training",
            "segments": [],
            "cumulative_iterations": 0,
            "selected_checkpoint": None,
        }
        campaign["runs"].append(seed_record)
        atomic_write_json(campaign_path, campaign)
        resume_checkpoint = None
        tail: list[Path] = []
        while True:
            cumulative = int(seed_record["cumulative_iterations"])
            iterations = next_training_segment_iterations(
                cumulative,
                len(seed_record["segments"]),
                initial_iterations,
                extension_iterations,
                max_total_iterations,
            )
            if iterations <= 0:
                last = seed_record["segments"][-1]
                raise RuntimeError(
                    f"seed={seed} reached max_total_iterations={max_total_iterations} "
                    f"without training eligibility; last_diagnostics="
                    f"{last.get('final_training_diagnostics')}; reasons="
                    f"{last.get('eligibility_failures')}"
                )
            run_dir, final_checkpoint, status = run_training_segment(
                int(seed), len(seed_record["segments"]), iterations, resume_checkpoint
            )
            seed_record["cumulative_iterations"] = cumulative + iterations
            segment = {
                "index": len(seed_record["segments"]),
                "run_dir": project_relative_path(run_dir),
                "requested_iterations": iterations,
                "cumulative_iterations": seed_record["cumulative_iterations"],
                "resume_checkpoint": (
                    project_relative_path(resume_checkpoint)
                    if resume_checkpoint is not None else None
                ),
                "final_checkpoint": project_relative_path(final_checkpoint),
                "training_eligible": status.get("training_eligible") is True,
                "eligibility_failures": status.get("eligibility_failures", []),
                "final_training_diagnostics": status.get(
                    "final_training_diagnostics"
                ),
            }
            seed_record["segments"].append(segment)
            atomic_write_json(campaign_path, campaign)
            tail = eligible_run_checkpoints(run_dir)
            if len(tail) == 5:
                break
            if not auto_extend:
                raise RuntimeError(
                    f"training run is not eligible for formal evaluation: {run_dir}; "
                    f"reasons={status.get('eligibility_failures')}"
                )
            if int(seed_record["cumulative_iterations"]) >= max_total_iterations:
                raise RuntimeError(
                    f"seed={seed} reached max_total_iterations={max_total_iterations} "
                    f"without training eligibility; last_diagnostics="
                    f"{status.get('final_training_diagnostics')}; reasons="
                    f"{status.get('eligibility_failures')}"
                )
            resume_checkpoint = final_checkpoint
            print(
                f"auto_extending seed={seed} cumulative_iterations="
                f"{seed_record['cumulative_iterations']} next_checkpoint="
                f"{resume_checkpoint}",
                flush=True,
            )
        validation = validate_seed_candidates(int(seed), tail)
        seed_record.update(
            {
                "status": "completed",
                "run_dir": project_relative_path(run_dir),
                "final_checkpoint": project_relative_path(final_checkpoint),
                "tail_checkpoints": [project_relative_path(path) for path in tail],
                "selected_checkpoint": validation["selected_checkpoint"],
                "validation": validation,
            }
        )
        campaign["checkpoints"].append(validation["selected_checkpoint"])
        atomic_write_json(campaign_path, campaign)
except BaseException as error:
    if campaign["runs"] and campaign["runs"][-1].get("status") == "training":
        campaign["runs"][-1]["status"] = "failed"
        campaign["runs"][-1]["failure"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
    campaign.update(
        {
            "status": "failed",
            "finished_at": datetime.now().astimezone().isoformat(),
            "failure": {"type": type(error).__name__, "message": str(error)},
        }
    )
    atomic_write_json(campaign_path, campaign)
    raise

campaign.update(
    {
        "status": "completed",
        "finished_at": datetime.now().astimezone().isoformat(),
        "failure": None,
    }
)
atomic_write_json(campaign_path, campaign)
atomic_write_json(
    campaign_root / "latest_campaign.json",
    {
        "schema_version": 1,
        "campaign": project_relative_path(campaign_path),
        "finished_at": campaign["finished_at"],
    },
)
print(f"campaign={campaign_path}", flush=True)
