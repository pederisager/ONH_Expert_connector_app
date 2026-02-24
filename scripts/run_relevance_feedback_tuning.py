from __future__ import annotations

import argparse
import copy
import json
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_USER64_BENCHMARK = Path("tests/benchmarks/search_relevance_chatgpt_user64_v1.yaml")
DEFAULT_STRICT100_BENCHMARK = Path("tests/benchmarks/search_relevance_100_v1.yaml")


@dataclass(slots=True)
class CommandResult:
    command: list[str]
    returncode: int
    duration_sec: float
    stdout_path: str
    stderr_path: str
    timed_out: bool = False


@dataclass(slots=True)
class TrialSpec:
    trial_id: str
    label: str
    scoring_weights: dict[str, float]
    exact_keyword_bonus: float
    category_base_penalty: float


@dataclass(slots=True)
class TrialOutcome:
    spec: TrialSpec
    user64_output: str | None
    strict100_output: str | None
    user64_metrics: dict[str, float] | None
    strict100_metrics: dict[str, float] | None
    user64_command: CommandResult | None
    strict100_command: CommandResult | None
    status: str
    score: float | None
    notes: list[str]


def _run_command(
    command: list[str],
    *,
    timeout_sec: int,
    stdout_path: Path,
    stderr_path: Path,
) -> CommandResult:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    timed_out = False
    with stdout_path.open("w", encoding="utf-8") as out_handle, stderr_path.open(
        "w", encoding="utf-8"
    ) as err_handle:
        try:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                timeout=max(1, int(timeout_sec)),
                check=False,
                stdout=out_handle,
                stderr=err_handle,
            )
            returncode = int(completed.returncode)
        except subprocess.TimeoutExpired:
            timed_out = True
            returncode = 124
    duration_sec = round(time.perf_counter() - started, 3)
    return CommandResult(
        command=command,
        returncode=returncode,
        duration_sec=duration_sec,
        stdout_path=str(stdout_path.relative_to(ROOT)).replace("\\", "/"),
        stderr_path=str(stderr_path.relative_to(ROOT)).replace("\\", "/"),
        timed_out=timed_out,
    )


def _wait_for_port(host: str, port: int, timeout_sec: int) -> bool:
    deadline = time.time() + max(1, int(timeout_sec))
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.settimeout(1.0)
            if probe.connect_ex((host, port)) == 0:
                return True
        time.sleep(0.5)
    return False


def _reserve_tcp_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind((host, 0))
        return int(probe.getsockname()[1])


def _wait_for_queue_probe(base_url: str, timeout_sec: int) -> bool:
    deadline = time.time() + max(1, timeout_sec)
    probe_url = f"{base_url.rstrip('/')}/queue"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(probe_url, timeout=2.0) as response:  # noqa: S310
                if int(getattr(response, "status", 0)) == 200:
                    return True
        except (urllib.error.URLError, TimeoutError, OSError):
            pass
        time.sleep(0.4)
    return False


def _run_benchmark_with_local_server(
    *,
    python_executable: str,
    benchmark_path: Path,
    output_path: Path,
    logs_dir: Path,
    trial_id: str,
    benchmark_slug: str,
    server_start_timeout_sec: int,
    benchmark_timeout_sec: int,
) -> CommandResult:
    server_port = _reserve_tcp_port("127.0.0.1")
    server_base_url = f"http://127.0.0.1:{server_port}"
    server_stdout = logs_dir / f"{trial_id}_{benchmark_slug}_uvicorn_stdout.log"
    server_stderr = logs_dir / f"{trial_id}_{benchmark_slug}_uvicorn_stderr.log"
    with server_stdout.open("w", encoding="utf-8") as out_handle, server_stderr.open(
        "w", encoding="utf-8"
    ) as err_handle:
        server = subprocess.Popen(
            [
                python_executable,
                "-m",
                "uvicorn",
                "app.main:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(server_port),
            ],
            cwd=ROOT,
            stdout=out_handle,
            stderr=err_handle,
        )
    try:
        if not _wait_for_port("127.0.0.1", server_port, timeout_sec=server_start_timeout_sec):
            return CommandResult(
                command=[python_executable, "-m", "uvicorn", "app.main:app", "--port", str(server_port)],
                returncode=1,
                duration_sec=0.0,
                stdout_path=str(server_stdout.relative_to(ROOT)).replace("\\", "/"),
                stderr_path=str(server_stderr.relative_to(ROOT)).replace("\\", "/"),
                timed_out=False,
            )
        if not _wait_for_queue_probe(server_base_url, timeout_sec=server_start_timeout_sec):
            return CommandResult(
                command=[python_executable, "-m", "uvicorn", "app.main:app", "--port", str(server_port)],
                returncode=1,
                duration_sec=0.0,
                stdout_path=str(server_stdout.relative_to(ROOT)).replace("\\", "/"),
                stderr_path=str(server_stderr.relative_to(ROOT)).replace("\\", "/"),
                timed_out=False,
            )

        benchmark_stdout = logs_dir / f"{trial_id}_{benchmark_slug}_stdout.log"
        benchmark_stderr = logs_dir / f"{trial_id}_{benchmark_slug}_stderr.log"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            python_executable,
            "scripts/run_search_benchmark.py",
            "--benchmark",
            str(benchmark_path.relative_to(ROOT)).replace("\\", "/"),
            "--base-url",
            server_base_url,
            "--output",
            str(output_path.relative_to(ROOT)).replace("\\", "/"),
        ]
        return _run_command(
            command,
            timeout_sec=benchmark_timeout_sec,
            stdout_path=benchmark_stdout,
            stderr_path=benchmark_stderr,
        )
    finally:
        server.terminate()
        try:
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server.kill()


def _load_metrics(path: Path) -> dict[str, float] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics") or {}
    parsed: dict[str, float] = {}
    for key in (
        "MustInclude@3",
        "ShouldInclude@10",
        "HardExcludeRate@10",
        "PublicationEvidencePassRate",
    ):
        value = metrics.get(key)
        if value is None:
            continue
        parsed[key] = float(value)
    return parsed


def _build_trial_specs(base_results: dict[str, Any], *, max_trials: int = 0) -> list[TrialSpec]:
    scoring = dict(base_results.get("scoring-weights") or {})
    baseline = TrialSpec(
        trial_id="T00",
        label="baseline",
        scoring_weights={
            "semantic": float(scoring.get("semantic", 1.0)),
            "keywords": float(scoring.get("keywords", 0.2)),
            "tags": float(scoring.get("tags", 0.25)),
            "methods": float(scoring.get("methods", 0.15)),
        },
        exact_keyword_bonus=float(
            (base_results.get("exact-keyword-promotion") or {}).get("keyword-bonus", 0.35)
        ),
        category_base_penalty=float(
            (base_results.get("category-intent-penalty") or {}).get("base-penalty", 0.08)
        ),
    )

    candidates: list[TrialSpec] = [
        baseline,
        TrialSpec(
            trial_id="T01",
            label="keyword_plus_tag_boost",
            scoring_weights={
                "semantic": baseline.scoring_weights["semantic"],
                "keywords": min(0.4, baseline.scoring_weights["keywords"] + 0.05),
                "tags": min(0.45, baseline.scoring_weights["tags"] + 0.05),
                "methods": baseline.scoring_weights["methods"],
            },
            exact_keyword_bonus=min(0.6, baseline.exact_keyword_bonus + 0.05),
            category_base_penalty=baseline.category_base_penalty,
        ),
        TrialSpec(
            trial_id="T02",
            label="keyword_stronger",
            scoring_weights={
                "semantic": baseline.scoring_weights["semantic"],
                "keywords": min(0.45, baseline.scoring_weights["keywords"] + 0.1),
                "tags": baseline.scoring_weights["tags"],
                "methods": baseline.scoring_weights["methods"],
            },
            exact_keyword_bonus=min(0.7, baseline.exact_keyword_bonus + 0.1),
            category_base_penalty=min(0.14, baseline.category_base_penalty + 0.02),
        ),
        TrialSpec(
            trial_id="T03",
            label="tag_stronger",
            scoring_weights={
                "semantic": baseline.scoring_weights["semantic"],
                "keywords": baseline.scoring_weights["keywords"],
                "tags": min(0.5, baseline.scoring_weights["tags"] + 0.1),
                "methods": baseline.scoring_weights["methods"],
            },
            exact_keyword_bonus=baseline.exact_keyword_bonus,
            category_base_penalty=baseline.category_base_penalty,
        ),
        TrialSpec(
            trial_id="T04",
            label="cross_disciplinary_relaxation",
            scoring_weights={
                "semantic": baseline.scoring_weights["semantic"],
                "keywords": baseline.scoring_weights["keywords"],
                "tags": baseline.scoring_weights["tags"],
                "methods": baseline.scoring_weights["methods"],
            },
            exact_keyword_bonus=baseline.exact_keyword_bonus,
            category_base_penalty=max(0.02, baseline.category_base_penalty - 0.03),
        ),
    ]

    deduped: list[TrialSpec] = []
    seen: set[tuple[float, float, float, float, float, float]] = set()
    for trial in candidates:
        signature = (
            round(trial.scoring_weights["semantic"], 4),
            round(trial.scoring_weights["keywords"], 4),
            round(trial.scoring_weights["tags"], 4),
            round(trial.scoring_weights["methods"], 4),
            round(trial.exact_keyword_bonus, 4),
            round(trial.category_base_penalty, 4),
        )
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append(trial)

    if max_trials > 0:
        return deduped[:max_trials]
    return deduped


def _patch_results_config(base_config: dict[str, Any], trial: TrialSpec) -> dict[str, Any]:
    patched = copy.deepcopy(base_config)
    results = dict(patched.get("results") or {})

    scoring_weights = dict(results.get("scoring-weights") or {})
    scoring_weights.update(trial.scoring_weights)
    results["scoring-weights"] = scoring_weights

    exact_keyword = dict(results.get("exact-keyword-promotion") or {})
    exact_keyword["keyword-bonus"] = trial.exact_keyword_bonus
    results["exact-keyword-promotion"] = exact_keyword

    category_intent = dict(results.get("category-intent-penalty") or {})
    category_intent["base-penalty"] = trial.category_base_penalty
    results["category-intent-penalty"] = category_intent

    patched["results"] = results
    return patched


def _trial_score(
    user64_metrics: dict[str, float] | None,
    strict100_metrics: dict[str, float] | None,
) -> float | None:
    if not user64_metrics:
        return None
    must = float(user64_metrics.get("MustInclude@3", 0.0))
    should = float(user64_metrics.get("ShouldInclude@10", 0.0))
    strict_guard = float((strict100_metrics or {}).get("HardExcludeRate@10", 0.0))
    strict_must = float((strict100_metrics or {}).get("MustInclude@3", 0.0))
    return round((must * 100.0) + (should * 35.0) + (strict_guard * 15.0) + (strict_must * 10.0), 6)


def _pick_best_trial(outcomes: list[TrialOutcome]) -> TrialOutcome | None:
    viable = [
        outcome
        for outcome in outcomes
        if outcome.status == "ok" and outcome.score is not None
    ]
    if not viable:
        return None
    viable.sort(
        key=lambda outcome: (
            float(outcome.score or 0.0),
            float((outcome.user64_metrics or {}).get("MustInclude@3", 0.0)),
            float((outcome.user64_metrics or {}).get("ShouldInclude@10", 0.0)),
        ),
        reverse=True,
    )
    return viable[0]


def _write_decision_memo(
    *,
    memo_path: Path,
    run_date: str,
    outcomes: list[TrialOutcome],
    best: TrialOutcome | None,
    dry_run: bool,
) -> None:
    lines = [
        f"# Relevance Feedback Tuning Memo ({run_date})",
        "",
        f"- Dry run: `{dry_run}`",
        "",
        "## Trial outcomes",
        "",
    ]
    for outcome in outcomes:
        lines.append(f"- **{outcome.spec.trial_id} {outcome.spec.label}**: `{outcome.status}`")
        lines.append(
            "  - weights: "
            f"semantic={outcome.spec.scoring_weights['semantic']}, "
            f"keywords={outcome.spec.scoring_weights['keywords']}, "
            f"tags={outcome.spec.scoring_weights['tags']}, "
            f"methods={outcome.spec.scoring_weights['methods']}"
        )
        lines.append(
            "  - bonuses: "
            f"exact_keyword_bonus={outcome.spec.exact_keyword_bonus}, "
            f"category_base_penalty={outcome.spec.category_base_penalty}"
        )
        if outcome.user64_metrics:
            lines.append(
                "  - user64: "
                f"MustInclude@3={outcome.user64_metrics.get('MustInclude@3')}, "
                f"ShouldInclude@10={outcome.user64_metrics.get('ShouldInclude@10')}"
            )
        if outcome.strict100_metrics:
            lines.append(
                "  - strict100: "
                f"MustInclude@3={outcome.strict100_metrics.get('MustInclude@3')}, "
                f"HardExcludeRate@10={outcome.strict100_metrics.get('HardExcludeRate@10')}"
            )
        if outcome.score is not None:
            lines.append(f"  - tuning score: {outcome.score}")
        for note in outcome.notes:
            lines.append(f"  - note: {note}")
    lines.append("")
    if best:
        lines.append("## Recommended trial")
        lines.append("")
        lines.append(f"- `{best.spec.trial_id}` {best.spec.label}")
        if best.score is not None:
            lines.append(f"- Score: {best.score}")
    else:
        lines.append("## Recommended trial")
        lines.append("")
        lines.append("- None (no successful benchmark trials yet)")
    lines.append("")

    memo_path.parent.mkdir(parents=True, exist_ok=True)
    memo_path.write_text("\n".join(lines), encoding="utf-8")


def _resolve_path(path: Path) -> Path:
    return (ROOT / path).resolve() if not path.is_absolute() else path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run relevance-feedback tuning sweeps against user64 + strict100 benchmarks.",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--app-config", type=Path, default=Path("data/app.config.yaml"))
    parser.add_argument("--run-date", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    parser.add_argument("--output-root", type=Path, default=Path("reports/relevance_tuning"))
    parser.add_argument("--user64-benchmark", type=Path, default=DEFAULT_USER64_BENCHMARK)
    parser.add_argument("--strict100-benchmark", type=Path, default=DEFAULT_STRICT100_BENCHMARK)
    parser.add_argument("--server-start-timeout-sec", type=int, default=120)
    parser.add_argument("--benchmark-timeout-sec", type=int, default=2400)
    parser.add_argument("--max-trials", type=int, default=0)
    parser.add_argument("--skip-strict100", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--apply-best", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    app_config_path = _resolve_path(args.app_config)
    user64_benchmark_path = _resolve_path(args.user64_benchmark)
    strict100_benchmark_path = _resolve_path(args.strict100_benchmark)
    missing_inputs: list[str] = []
    if not app_config_path.exists():
        missing_inputs.append(f"app config: {app_config_path}")
    if not user64_benchmark_path.exists():
        missing_inputs.append(f"user64 benchmark: {user64_benchmark_path}")
    if not args.skip_strict100 and not strict100_benchmark_path.exists():
        missing_inputs.append(f"strict100 benchmark: {strict100_benchmark_path}")
    if missing_inputs:
        for missing in missing_inputs:
            print(f"[tuning] missing required input: {missing}", file=sys.stderr)
        return 2

    output_dir = (ROOT / args.output_root / args.run_date).resolve()
    logs_dir = output_dir / "logs"
    configs_dir = output_dir / "configs"
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    configs_dir.mkdir(parents=True, exist_ok=True)

    original_text = app_config_path.read_text(encoding="utf-8")
    base_config = yaml.safe_load(original_text) or {}
    base_results = dict(base_config.get("results") or {})
    trials = _build_trial_specs(base_results, max_trials=max(0, int(args.max_trials)))

    outcomes: list[TrialOutcome] = []

    try:
        for trial in trials:
            notes: list[str] = []
            patched_config = _patch_results_config(base_config, trial)
            trial_config_path = configs_dir / f"{trial.trial_id}.app.config.yaml"
            trial_config_path.write_text(
                yaml.safe_dump(patched_config, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )

            if args.dry_run:
                outcomes.append(
                    TrialOutcome(
                        spec=trial,
                        user64_output=None,
                        strict100_output=None,
                        user64_metrics=None,
                        strict100_metrics=None,
                        user64_command=None,
                        strict100_command=None,
                        status="dry_run",
                        score=None,
                        notes=["dry_run: benchmarks not executed"],
                    )
                )
                continue

            app_config_path.write_text(
                yaml.safe_dump(patched_config, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )

            user64_output = output_dir / f"{trial.trial_id}_user64.json"
            strict100_output = output_dir / f"{trial.trial_id}_strict100.json"

            user64_cmd = _run_benchmark_with_local_server(
                python_executable=args.python,
                benchmark_path=user64_benchmark_path,
                output_path=user64_output,
                logs_dir=logs_dir,
                trial_id=trial.trial_id,
                benchmark_slug="user64",
                server_start_timeout_sec=args.server_start_timeout_sec,
                benchmark_timeout_sec=args.benchmark_timeout_sec,
            )
            user64_metrics = _load_metrics(user64_output)

            strict100_cmd: CommandResult | None = None
            strict100_metrics: dict[str, float] | None = None
            status = "ok"
            if user64_cmd.returncode != 0:
                status = "user64_failed"
                notes.append("user64 benchmark failed")

            if status == "ok" and not args.skip_strict100:
                strict100_cmd = _run_benchmark_with_local_server(
                    python_executable=args.python,
                    benchmark_path=strict100_benchmark_path,
                    output_path=strict100_output,
                    logs_dir=logs_dir,
                    trial_id=trial.trial_id,
                    benchmark_slug="strict100",
                    server_start_timeout_sec=args.server_start_timeout_sec,
                    benchmark_timeout_sec=args.benchmark_timeout_sec,
                )
                strict100_metrics = _load_metrics(strict100_output)
                if strict100_cmd.returncode != 0:
                    status = "strict100_failed"
                    notes.append("strict100 benchmark failed")

            score = _trial_score(user64_metrics, strict100_metrics) if status == "ok" else None
            outcomes.append(
                TrialOutcome(
                    spec=trial,
                    user64_output=str(user64_output.relative_to(ROOT)).replace("\\", "/") if user64_output.exists() else None,
                    strict100_output=(
                        str(strict100_output.relative_to(ROOT)).replace("\\", "/")
                        if strict100_output.exists()
                        else None
                    ),
                    user64_metrics=user64_metrics,
                    strict100_metrics=strict100_metrics,
                    user64_command=user64_cmd,
                    strict100_command=strict100_cmd,
                    status=status,
                    score=score,
                    notes=notes,
                )
            )
    finally:
        app_config_path.write_text(original_text, encoding="utf-8")

    best = _pick_best_trial(outcomes)

    if args.apply_best and best and not args.dry_run:
        best_config = _patch_results_config(base_config, best.spec)
        app_config_path.write_text(
            yaml.safe_dump(best_config, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_date": args.run_date,
        "dry_run": bool(args.dry_run),
        "trials": [asdict(outcome) for outcome in outcomes],
        "best_trial_id": best.spec.trial_id if best else None,
        "best_trial_label": best.spec.label if best else None,
        "apply_best": bool(args.apply_best and best and not args.dry_run),
    }

    summary_path = output_dir / "tuning_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    memo_path = output_dir / "decision_memo.md"
    _write_decision_memo(
        memo_path=memo_path,
        run_date=args.run_date,
        outcomes=outcomes,
        best=best,
        dry_run=bool(args.dry_run),
    )

    print(f"[tuning] wrote summary: {summary_path.relative_to(ROOT)}")
    print(f"[tuning] wrote memo: {memo_path.relative_to(ROOT)}")

    failures = [outcome for outcome in outcomes if outcome.status not in {"ok", "dry_run"}]
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
