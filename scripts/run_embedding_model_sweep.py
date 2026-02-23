from __future__ import annotations

import argparse
import json
import os
import re
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
DEFAULT_MODELS = [
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "intfloat/multilingual-e5-large",
    "BAAI/bge-m3",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
]


@dataclass(slots=True)
class CommandResult:
    command: list[str]
    returncode: int
    duration_sec: float
    stdout_path: str
    stderr_path: str
    timed_out: bool = False


@dataclass(slots=True)
class BenchmarkMetrics:
    must_include_at_3: float | None = None
    should_include_at_10: float | None = None
    hard_exclude_rate_at_10: float | None = None
    publication_evidence_pass_rate: float | None = None


@dataclass(slots=True)
class CandidateResult:
    model_name: str
    model_slug: str
    build: CommandResult | None
    user64: CommandResult | None
    strict100: CommandResult | None
    user64_metrics: BenchmarkMetrics | None
    strict100_metrics: BenchmarkMetrics | None
    gpu_before: dict[str, Any] | None
    gpu_after: dict[str, Any] | None
    status: str
    notes: list[str]


def _slugify(value: str) -> str:
    collapsed = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    return collapsed.strip("-").lower() or "model"


def _run_command(
    command: list[str],
    *,
    cwd: Path,
    timeout_sec: int,
    stdout_path: Path,
    stderr_path: Path,
    env: dict[str, str] | None = None,
) -> CommandResult:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    timed_out = False
    with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr_handle:
        try:
            completed = subprocess.run(
                command,
                cwd=cwd,
                timeout=max(1, int(timeout_sec)),
                check=False,
                stdout=stdout_handle,
                stderr=stderr_handle,
                env=env,
            )
            returncode = int(completed.returncode)
        except subprocess.TimeoutExpired:
            timed_out = True
            returncode = 124
    duration = round(time.perf_counter() - started, 3)
    return CommandResult(
        command=command,
        returncode=returncode,
        duration_sec=duration,
        stdout_path=str(stdout_path.relative_to(ROOT)).replace("\\", "/"),
        stderr_path=str(stderr_path.relative_to(ROOT)).replace("\\", "/"),
        timed_out=timed_out,
    )


def _reserve_tcp_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind((host, 0))
        return int(probe.getsockname()[1])


def _wait_for_port(host: str, port: int, timeout_sec: int) -> bool:
    deadline = time.time() + max(1, timeout_sec)
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.settimeout(1.0)
            if probe.connect_ex((host, port)) == 0:
                return True
        time.sleep(0.5)
    return False


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


def _load_benchmark_payload(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _load_benchmark_metrics(path: Path) -> BenchmarkMetrics | None:
    payload = _load_benchmark_payload(path)
    if not payload:
        return None
    metrics = payload.get("metrics") or {}
    return BenchmarkMetrics(
        must_include_at_3=_as_float(metrics.get("MustInclude@3")),
        should_include_at_10=_as_float(metrics.get("ShouldInclude@10")),
        hard_exclude_rate_at_10=_as_float(metrics.get("HardExcludeRate@10")),
        publication_evidence_pass_rate=_as_float(metrics.get("PublicationEvidencePassRate")),
    )


def _benchmark_request_error_count(payload: dict[str, Any] | None) -> int:
    if not payload:
        return 0
    count = 0
    for row in payload.get("query_results") or []:
        if isinstance(row, dict) and row.get("request_error"):
            count += 1
    return count


def _benchmark_failure_count(payload: dict[str, Any] | None) -> int:
    if not payload:
        return 0
    return (
        len(payload.get("threshold_failures") or [])
        + len(payload.get("mode_threshold_failures") or [])
        + len(payload.get("overexposure_violations") or [])
    )


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _query_gpu_snapshot() -> dict[str, Any] | None:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None

    first_line = (output.splitlines() or [""])[0]
    parts = [part.strip() for part in first_line.split(",")]
    if len(parts) < 4:
        return {"raw": first_line}
    return {
        "name": parts[0],
        "memory_total_mb": _as_float(parts[1]),
        "memory_used_mb": _as_float(parts[2]),
        "gpu_utilization_pct": _as_float(parts[3]),
    }


def _write_models_config(
    models_config_path: Path,
    *,
    model_name: str,
    device: str,
) -> None:
    payload = yaml.safe_load(models_config_path.read_text(encoding="utf-8")) or {}
    embedding = dict(payload.get("embedding_model") or {})
    embedding["name"] = model_name
    embedding["backend"] = "sentence_transformers"
    embedding["device"] = device
    payload["embedding_model"] = embedding
    models_config_path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _run_benchmark_with_local_server(
    *,
    python_executable: str,
    benchmark_path: Path,
    output_path: Path,
    logs_dir: Path,
    model_slug: str,
    benchmark_slug: str,
    start_timeout_sec: int,
    benchmark_timeout_sec: int,
) -> CommandResult:
    server_port = _reserve_tcp_port("127.0.0.1")
    server_base_url = f"http://127.0.0.1:{server_port}"
    server_stdout = logs_dir / f"{model_slug}_uvicorn_stdout.log"
    server_stderr = logs_dir / f"{model_slug}_uvicorn_stderr.log"
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
            env=dict(os.environ),
        )
    try:
        if not _wait_for_port("127.0.0.1", server_port, timeout_sec=start_timeout_sec):
            return CommandResult(
                command=[python_executable, "-m", "uvicorn", "app.main:app", "--port", str(server_port)],
                returncode=1,
                duration_sec=0.0,
                stdout_path=str(server_stdout.relative_to(ROOT)).replace("\\", "/"),
                stderr_path=str(server_stderr.relative_to(ROOT)).replace("\\", "/"),
                timed_out=False,
            )
        if not _wait_for_queue_probe(server_base_url, timeout_sec=start_timeout_sec):
            return CommandResult(
                command=[python_executable, "-m", "uvicorn", "app.main:app", "--port", str(server_port)],
                returncode=1,
                duration_sec=0.0,
                stdout_path=str(server_stdout.relative_to(ROOT)).replace("\\", "/"),
                stderr_path=str(server_stderr.relative_to(ROOT)).replace("\\", "/"),
                timed_out=False,
            )
        bench_stdout = logs_dir / f"{model_slug}_{benchmark_slug}_stdout.log"
        bench_stderr = logs_dir / f"{model_slug}_{benchmark_slug}_stderr.log"
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
            cwd=ROOT,
            timeout_sec=benchmark_timeout_sec,
            stdout_path=bench_stdout,
            stderr_path=bench_stderr,
        )
    finally:
        server.terminate()
        try:
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server.kill()


def _choose_best_model(results: list[CandidateResult]) -> str | None:
    scored: list[tuple[float, float, str]] = []
    for result in results:
        if result.status != "ok" or not result.user64_metrics:
            continue
        must = result.user64_metrics.must_include_at_3
        should = result.user64_metrics.should_include_at_10
        if must is None:
            continue
        scored.append((must, should or 0.0, result.model_name))
    if not scored:
        return None
    scored.sort(reverse=True)
    return scored[0][2]


def _write_decision_memo(
    *,
    memo_path: Path,
    run_date: str,
    results: list[CandidateResult],
    recommended_model: str | None,
) -> None:
    lines = [
        f"# Model Sweep Decision Memo ({run_date})",
        "",
        "## Candidate outcomes",
        "",
    ]
    for result in results:
        lines.append(f"- **{result.model_name}**: `{result.status}`")
        if result.user64_metrics:
            lines.append(
                "  - user64 MustInclude@3="
                f"{result.user64_metrics.must_include_at_3}"
                ", ShouldInclude@10="
                f"{result.user64_metrics.should_include_at_10}"
            )
        if result.strict100_metrics:
            lines.append(
                "  - strict100 MustInclude@3="
                f"{result.strict100_metrics.must_include_at_3}"
                ", HardExcludeRate@10="
                f"{result.strict100_metrics.hard_exclude_rate_at_10}"
            )
        if result.notes:
            for note in result.notes:
                lines.append(f"  - note: {note}")
    lines.append("")
    if recommended_model:
        lines.append(f"## Recommended model\n\n- {recommended_model}")
    else:
        lines.append("## Recommended model\n\n- No recommendation yet (insufficient successful runs).")
    lines.append("")
    memo_path.parent.mkdir(parents=True, exist_ok=True)
    memo_path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run embedding model sweep with index rebuild + user64/strict100 benchmarks.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Embedding model names to evaluate.",
    )
    parser.add_argument(
        "--models-config",
        type=Path,
        default=Path("data/models.yaml"),
        help="Path to models.yaml.",
    )
    parser.add_argument(
        "--run-date",
        default=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        help="Date folder under reports/model_sweeps (default: today UTC).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("reports/model_sweeps"),
        help="Root output directory for sweep reports.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable for build/server/benchmark commands.",
    )
    parser.add_argument(
        "--index-timeout-sec",
        type=int,
        default=2400,
    )
    parser.add_argument(
        "--benchmark-timeout-sec",
        type=int,
        default=2400,
    )
    parser.add_argument(
        "--server-start-timeout-sec",
        type=int,
        default=120,
    )
    parser.add_argument(
        "--benchmark-retries",
        type=int,
        default=1,
        help="Retry count when benchmark output contains request errors (default: 1).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Embedding device to write in models.yaml (default: auto).",
    )
    parser.add_argument(
        "--user64-benchmark",
        type=Path,
        default=DEFAULT_USER64_BENCHMARK,
    )
    parser.add_argument(
        "--strict100-benchmark",
        type=Path,
        default=DEFAULT_STRICT100_BENCHMARK,
    )
    parser.add_argument(
        "--max-models",
        type=int,
        default=0,
        help="Optional cap on number of models to execute (0 = all).",
    )
    parser.add_argument("--skip-index", action="store_true")
    parser.add_argument("--skip-benchmarks", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    models_path = (ROOT / args.models_config).resolve() if not args.models_config.is_absolute() else args.models_config
    output_dir = (ROOT / args.output_root / args.run_date).resolve()
    logs_dir = output_dir / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    selected_models = list(args.models)
    if args.max_models and args.max_models > 0:
        selected_models = selected_models[: args.max_models]

    original_models_yaml = models_path.read_text(encoding="utf-8")
    results: list[CandidateResult] = []

    try:
        for model_name in selected_models:
            model_slug = _slugify(model_name)
            notes: list[str] = []
            print(f"[sweep] model={model_name}")
            gpu_before = _query_gpu_snapshot()

            if args.dry_run:
                notes.append("dry_run: no build/benchmark commands executed")
                results.append(
                    CandidateResult(
                        model_name=model_name,
                        model_slug=model_slug,
                        build=None,
                        user64=None,
                        strict100=None,
                        user64_metrics=None,
                        strict100_metrics=None,
                        gpu_before=gpu_before,
                        gpu_after=_query_gpu_snapshot(),
                        status="dry_run",
                        notes=notes,
                    )
                )
                continue

            _write_models_config(models_path, model_name=model_name, device=args.device)

            build_result: CommandResult | None = None
            user64_result: CommandResult | None = None
            strict_result: CommandResult | None = None
            user64_metrics: BenchmarkMetrics | None = None
            strict_metrics: BenchmarkMetrics | None = None
            status = "ok"

            if not args.skip_index:
                build_result = _run_command(
                    [
                        args.python,
                        "-m",
                        "app.index.build",
                        "--nva-results",
                        "data/nva/results.jsonl",
                    ],
                    cwd=ROOT,
                    timeout_sec=args.index_timeout_sec,
                    stdout_path=logs_dir / f"{model_slug}_build_stdout.log",
                    stderr_path=logs_dir / f"{model_slug}_build_stderr.log",
                )
                if build_result.returncode != 0:
                    status = "build_failed"
                    notes.append("index build failed")

            if status == "ok" and not args.skip_benchmarks:
                benchmark_retries = max(0, int(args.benchmark_retries))
                user64_output = output_dir / f"{model_slug}_user64.json"
                strict_output = output_dir / f"{model_slug}_strict100.json"

                user64_result = _run_benchmark_with_local_server(
                    python_executable=args.python,
                    benchmark_path=(ROOT / args.user64_benchmark).resolve()
                    if not args.user64_benchmark.is_absolute()
                    else args.user64_benchmark,
                    output_path=user64_output,
                    logs_dir=logs_dir,
                    model_slug=model_slug,
                    benchmark_slug="user64",
                    start_timeout_sec=args.server_start_timeout_sec,
                    benchmark_timeout_sec=args.benchmark_timeout_sec,
                )
                user64_payload = _load_benchmark_payload(user64_output)
                user64_request_errors = _benchmark_request_error_count(user64_payload)
                retry_index = 0
                while user64_request_errors > 0 and retry_index < benchmark_retries:
                    retry_index += 1
                    notes.append(
                        "user64 retry "
                        f"{retry_index}/{benchmark_retries} after {user64_request_errors} request errors"
                    )
                    user64_result = _run_benchmark_with_local_server(
                        python_executable=args.python,
                        benchmark_path=(ROOT / args.user64_benchmark).resolve()
                        if not args.user64_benchmark.is_absolute()
                        else args.user64_benchmark,
                        output_path=user64_output,
                        logs_dir=logs_dir,
                        model_slug=model_slug,
                        benchmark_slug="user64",
                        start_timeout_sec=args.server_start_timeout_sec,
                        benchmark_timeout_sec=args.benchmark_timeout_sec,
                    )
                    user64_payload = _load_benchmark_payload(user64_output)
                    user64_request_errors = _benchmark_request_error_count(user64_payload)

                user64_metrics = _load_benchmark_metrics(user64_output)
                if user64_result.returncode != 0:
                    if user64_request_errors > 0:
                        status = "user64_infra_failed"
                        notes.append(
                            f"user64 benchmark ended with {user64_request_errors} request errors"
                        )
                    elif _benchmark_failure_count(user64_payload) > 0:
                        status = "user64_threshold_failed"
                        notes.append("user64 benchmark failed thresholds")
                    else:
                        status = "user64_failed"
                        notes.append("user64 benchmark failed")

                if status == "ok":
                    strict_result = _run_benchmark_with_local_server(
                        python_executable=args.python,
                        benchmark_path=(ROOT / args.strict100_benchmark).resolve()
                        if not args.strict100_benchmark.is_absolute()
                        else args.strict100_benchmark,
                        output_path=strict_output,
                        logs_dir=logs_dir,
                        model_slug=model_slug,
                        benchmark_slug="strict100",
                        start_timeout_sec=args.server_start_timeout_sec,
                        benchmark_timeout_sec=args.benchmark_timeout_sec,
                    )
                    strict_payload = _load_benchmark_payload(strict_output)
                    strict_request_errors = _benchmark_request_error_count(strict_payload)
                    retry_index = 0
                    while strict_request_errors > 0 and retry_index < benchmark_retries:
                        retry_index += 1
                        notes.append(
                            "strict100 retry "
                            f"{retry_index}/{benchmark_retries} after {strict_request_errors} request errors"
                        )
                        strict_result = _run_benchmark_with_local_server(
                            python_executable=args.python,
                            benchmark_path=(ROOT / args.strict100_benchmark).resolve()
                            if not args.strict100_benchmark.is_absolute()
                            else args.strict100_benchmark,
                            output_path=strict_output,
                            logs_dir=logs_dir,
                            model_slug=model_slug,
                            benchmark_slug="strict100",
                            start_timeout_sec=args.server_start_timeout_sec,
                            benchmark_timeout_sec=args.benchmark_timeout_sec,
                        )
                        strict_payload = _load_benchmark_payload(strict_output)
                        strict_request_errors = _benchmark_request_error_count(strict_payload)

                    strict_metrics = _load_benchmark_metrics(strict_output)
                    if strict_result.returncode != 0:
                        if strict_request_errors > 0:
                            status = "strict100_infra_failed"
                            notes.append(
                                f"strict100 benchmark ended with {strict_request_errors} request errors"
                            )
                        elif _benchmark_failure_count(strict_payload) > 0:
                            status = "strict100_threshold_failed"
                            notes.append("strict100 benchmark failed thresholds")
                        else:
                            status = "strict100_failed"
                            notes.append("strict100 benchmark failed")

            results.append(
                CandidateResult(
                    model_name=model_name,
                    model_slug=model_slug,
                    build=build_result,
                    user64=user64_result,
                    strict100=strict_result,
                    user64_metrics=user64_metrics,
                    strict100_metrics=strict_metrics,
                    gpu_before=gpu_before,
                    gpu_after=_query_gpu_snapshot(),
                    status=status,
                    notes=notes,
                )
            )
    finally:
        models_path.write_text(original_models_yaml, encoding="utf-8")

    summary_path = output_dir / "sweep_summary.json"
    summary_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_date": args.run_date,
        "models": [asdict(result) for result in results],
        "recommended_model": _choose_best_model(results),
    }
    summary_path.write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    memo_path = output_dir / "decision_memo.md"
    _write_decision_memo(
        memo_path=memo_path,
        run_date=args.run_date,
        results=results,
        recommended_model=summary_payload["recommended_model"],
    )

    print(f"[sweep] wrote summary: {summary_path.relative_to(ROOT)}")
    print(f"[sweep] wrote memo: {memo_path.relative_to(ROOT)}")

    failures = [result for result in results if result.status not in {"ok", "dry_run"}]
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
