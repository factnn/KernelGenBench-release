#!/usr/bin/env python3
"""Re-verify a corpus of saved kernels under a validation policy.

This driver exists so that the paper's correctness oracle and performance
protocol can be stress-tested *without re-invoking any model*: it takes kernels
that the pipeline already accepted and re-runs the real verification path on
them under an alternative policy.

Policies (see ``src/sandbox/utils/accuracy_utils.py`` for the switches):

  baseline     published protocol (atol = 1e-4 * D_reduce, published shapes)
  heldout      published shapes + held-out shapes/strides and a fresh seed

Usage:
  python scripts/analyze/reverify_corpus.py \
      --kernels <dir> --policy baseline --out runs/baseline.json --jobs 8
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
VERIFY_CLI = REPO / "agent_bench" / "tools" / "verify_single.py"

# All three sources live in separate test modules; the original evaluation
# protocol loaded all of them (see agent_bench/config.example.yaml).
ALL_TEST_MODULES = ",".join([
    "src/kernelgenbench/accuracy/test_ops_with_benchmark.py",
    "src/kernelgenbench/accuracy/cublas",
    "src/kernelgenbench/accuracy/vllm13",
])

POLICIES = {
    "baseline": {},
    "heldout": {"KGB_HELDOUT": "1", "KGB_SEED_OFFSET": "1000"},
}

# file names are "<namespace>__<operator>.py"
def parse_kernel_name(path: Path):
    stem = path.name[:-3] if path.name.endswith(".py") else path.name
    if "__" not in stem:
        return None
    namespace, operator = stem.split("__", 1)
    if not namespace or not operator:
        return None
    return namespace, operator


def run_one(item, policy_env, python, gpu, timeout, extra_env):
    kernel_path, namespace, operator = item
    env = os.environ.copy()
    for name in ("KGB_HELDOUT", "KGB_SEED_OFFSET",
                 "KGB_AUDIT", "KGB_VERIFY_DIR"):
        env.pop(name, None)
    env.update(policy_env)
    env.update(extra_env)
    env["PYTHONPATH"] = str(REPO / "src")
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    safe = f"{namespace}__{operator}".replace("/", "_")
    audit_path = env.get("KGB_AUDIT")
    if audit_path:
        # one audit file per operator so that concurrent workers never interleave
        env["KGB_AUDIT"] = str(Path(audit_path) / f"{safe}.jsonl")
        Path(env["KGB_AUDIT"]).parent.mkdir(parents=True, exist_ok=True)
    verify_root = env.get("KGB_VERIFY_DIR")
    verify_dir = None
    if verify_root:
        # the verifier only reports a speedup when it persists its logs
        verify_dir = Path(verify_root) / safe
        verify_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        python,
        str(VERIFY_CLI),
        "--code", str(kernel_path),
        "--op", operator,
        "--namespace", namespace,
        "--test-modules", ALL_TEST_MODULES,
        "--timeout", str(timeout),
        "--output-json",
    ]
    if verify_dir is not None:
        cmd += ["--output-dir", str(verify_dir)]
    started = time.time()
    try:
        proc = subprocess.run(
            cmd, cwd=str(REPO), env=env, capture_output=True, text=True,
            timeout=timeout + 300,
        )
        stdout = proc.stdout.strip().splitlines()
        payload = None
        for line in reversed(stdout):
            line = line.strip()
            if line.startswith("{"):
                try:
                    payload = json.loads(line)
                    break
                except json.JSONDecodeError:
                    continue
        if payload is None:
            payload = {
                "passed": False,
                "error": f"no JSON result; rc={proc.returncode}; "
                         f"stderr_tail={proc.stderr[-400:]}",
                "total_tests": 0, "passed_tests": 0, "failed_tests": 0,
            }
        payload["returncode"] = proc.returncode
    except subprocess.TimeoutExpired:
        payload = {
            "passed": False, "error": "driver timeout", "total_tests": 0,
            "passed_tests": 0, "failed_tests": 0, "returncode": None,
        }
    if verify_dir is not None:
        verify_json = verify_dir / "verify.json"
        if verify_json.exists():
            try:
                saved = json.loads(verify_json.read_text())
                if isinstance(saved.get("speedup"), (int, float)):
                    payload["speedup"] = saved["speedup"]
            except (json.JSONDecodeError, OSError):
                pass

    payload["op"] = f"{namespace}::{operator}"
    payload["kernel"] = str(kernel_path)
    payload["wall_s"] = round(time.time() - started, 1)

    # Summarise the tolerance audit for this operator, if one was produced.
    if audit_path:
        path = Path(env["KGB_AUDIT"])
        if path.exists():
            checks = []
            for line in path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("kind") == "tolerance":
                    checks.append(rec)
            if checks:
                import math
                worst = max(checks, key=lambda r: r["max_slack"])
                payload["tolerance"] = {
                    "n_checks": len(checks),
                    "max_slack": worst["max_slack"],
                    "worst_reduce_dim": worst["reduce_dim"],
                    "published_atol": 1e-4 * max(worst["reduce_dim"], 1),
                    # a rule passes iff every recorded comparison needs at most
                    # that rule's atol
                    "passes_const": all(c["max_slack"] <= 1e-4 for c in checks),
                    "passes_sqrt": all(
                        c["max_slack"] <= 1e-4 * math.sqrt(max(c["reduce_dim"], 1))
                        for c in checks
                    ),
                    "passes_scaled": all(
                        c["max_slack"] <= 1e-4 * max(c["reduce_dim"], 1)
                        for c in checks
                    ),
                    "max_reduce_dim": max(c["reduce_dim"] for c in checks),
                }
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernels", required=True,
                    help="directory of <namespace>__<op>.py kernels")
    ap.add_argument("--policy", required=True, choices=sorted(POLICIES))
    ap.add_argument("--out", required=True, help="output JSON path")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7")
    ap.add_argument("--timeout", type=int, default=900,
                    help="per-operator accuracy timeout passed to the verifier")
    ap.add_argument("--only", type=str, default=None,
                    help="comma separated list of full op names to restrict to")
    ap.add_argument("--verify-dir", type=str, default=None,
                    help="persist verifier logs here so that per-operator "
                         "speedups are available in the results")
    ap.add_argument("--audit-dir", type=str, default=None,
                    help="collect per-comparison tolerance slack into this "
                         "directory (one JSONL per operator)")
    ap.add_argument("--python", type=str,
                    default=sys.executable or "python3")
    args = ap.parse_args()

    kernel_dir = Path(args.kernels).resolve()
    items = []
    for path in sorted(kernel_dir.glob("*.py")):
        parsed = parse_kernel_name(path)
        if parsed is None:
            continue
        items.append((path, parsed[0], parsed[1]))
    if args.only:
        wanted = {s.strip() for s in args.only.split(",") if s.strip()}
        items = [it for it in items if f"{it[1]}::{it[2]}" in wanted]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if out_path.exists():
        try:
            prev = json.loads(out_path.read_text())
            done = {r["op"]: r for r in prev.get("results", []) if r.get("op")}
            print(f"[resume] {len(done)} operators already present")
        except Exception:
            done = {}
    items = [it for it in items if f"{it[1]}::{it[2]}" not in done]

    gpus = [g for g in args.gpus.split(",") if g != ""]
    policy_env = dict(POLICIES[args.policy])
    if args.audit_dir:
        Path(args.audit_dir).mkdir(parents=True, exist_ok=True)
        policy_env["KGB_AUDIT"] = args.audit_dir
    if args.verify_dir:
        Path(args.verify_dir).mkdir(parents=True, exist_ok=True)
        policy_env["KGB_VERIFY_DIR"] = args.verify_dir
    print(f"[policy] {args.policy} env={policy_env}")
    print(f"[corpus] {len(items)} operators to verify on {len(gpus)} GPUs")

    results = list(done.values())
    if items:
        with ThreadPoolExecutor(max_workers=min(args.jobs, len(gpus))) as pool:
            available_gpus = Queue()
            for gpu in gpus:
                available_gpus.put(gpu)
            def run_exclusive(item):
                gpu = available_gpus.get()
                try:
                    return run_one(item, policy_env, args.python, gpu, args.timeout, {})
                finally:
                    available_gpus.put(gpu)
            futures = {pool.submit(run_exclusive, item): item for item in items}
            for n, fut in enumerate(as_completed(futures), 1):
                res = fut.result()
                results.append(res)
                status = "PASS" if res.get("passed") else "FAIL"
                print(f"[{n}/{len(items)}] {status} {res['op']} "
                      f"({res.get('passed_tests')}/{res.get('total_tests')}) "
                      f"{res.get('wall_s')}s", flush=True)
                doc = {
                    "policy": args.policy,
                    "policy_env": policy_env,
                    "corpus": str(kernel_dir),
                    "results": sorted(results, key=lambda r: r["op"]),
                }
                out_path.write_text(json.dumps(doc, indent=2))

    passed = sum(1 for r in results if r.get("passed"))
    print(f"\n=== {args.policy}: {passed}/{len(results)} passed ===")
    out_path.write_text(json.dumps({
        "policy": args.policy,
        "policy_env": policy_env,
        "corpus": str(kernel_dir),
        "results": sorted(results, key=lambda r: r["op"]),
    }, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
