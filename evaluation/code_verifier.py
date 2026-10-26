#!/usr/bin/env python3
"""Isolated verifier for APPS-style code-generation records.

The parent process never executes candidate code.  A fresh isolated Python child
runs an entire deterministic test selection with CPU/address-space/file/process
limits.  Both APPS stdin/stdout and call-based records are supported.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, re, resource, signal, subprocess, sys, tempfile, time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

_SAFE_OS_OPEN, _SAFE_OS_WRITE, _SAFE_OS_CLOSE = os.open, os.write, os.close

_LAST_BLOCK = re.compile(r"```(?:python|py)?\s*\n?(.*?)```", re.I | re.S)


def extract_code(text: str) -> str:
    blocks = _LAST_BLOCK.findall(text or "")
    if blocks:
        return blocks[-1].strip()
    return (text or "").strip()


def deterministic_indices(n: int, limit: int, seed_key: str = "") -> List[int]:
    if limit <= 0 or n <= limit:
        return list(range(n))
    # Stable spread including both endpoints; hash only rotates interior ties.
    if limit == 1:
        return [int(hashlib.sha256(seed_key.encode()).hexdigest(), 16) % n]
    vals = [round(i * (n - 1) / (limit - 1)) for i in range(limit)]
    return sorted(set(vals))


def _limit_child(cpu_seconds: int, memory_mb: int, file_mb: int, nofile: int, nproc: int) -> None:
    cpu = max(1, int(cpu_seconds))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))
    mem = int(memory_mb) * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
    resource.setrlimit(resource.RLIMIT_FSIZE, (int(file_mb) * 1024 * 1024,) * 2)
    resource.setrlimit(resource.RLIMIT_NOFILE, (int(nofile),) * 2)
    try:
        resource.setrlimit(resource.RLIMIT_NPROC, (int(nproc),) * 2)
    except (ValueError, OSError):
        pass
    os.setsid()


def verify_code(code: str, gold: Dict[str, Any], timeout: float = 10.0,
                max_tests: int = 0, seed_key: str = "", memory_mb: int = 2048) -> Dict[str, Any]:
    started = time.time()
    inputs, outputs = gold.get("inputs"), gold.get("outputs")
    if not isinstance(inputs, list) or not isinstance(outputs, list) or not inputs or len(inputs) != len(outputs):
        return {"status": "invalid_tests", "passed_tests": 0, "total_tests": 0,
                "elapsed_sec": time.time() - started, "per_test": []}
    idx = deterministic_indices(len(inputs), int(max_tests), seed_key)
    selected = {"kind": gold.get("kind"), "fn_name": gold.get("fn_name"),
                "inputs": [inputs[i] for i in idx], "outputs": [outputs[i] for i in idx],
                "original_indices": idx}
    with tempfile.TemporaryDirectory(prefix="pmopd_code_") as td:
        req = Path(td) / "request.json"
        outp = Path(td) / "result.json"
        req.write_text(json.dumps({"code": code, "gold": selected}, ensure_ascii=False), encoding="utf-8")
        worker = Path(__file__).resolve()
        # Per-test signal timers exist in the worker; outer wall cap handles blocked C calls.
        wall = max(3.0, min(180.0, timeout * len(idx) + 5.0))
        cpu = max(2, min(180, math.ceil(timeout * len(idx)) + 2))
        cmd = [sys.executable, "-I", str(worker), "--worker", str(req), str(outp), "--timeout", str(timeout)]
        proc = None
        try:
            proc = subprocess.Popen(cmd, cwd=td, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, text=True,
                                    preexec_fn=lambda: _limit_child(cpu, memory_mb, 8, 64, 0))
            stdout, stderr = proc.communicate(timeout=wall)
        except subprocess.TimeoutExpired:
            if proc is not None:
                try: os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError: pass
                proc.communicate()
            return {"status": "timeout", "passed_tests": 0, "total_tests": len(idx),
                    "elapsed_sec": time.time() - started, "per_test": [], "selected_indices": idx}
        except Exception as exc:
            return {"status": "sandbox_error", "passed_tests": 0, "total_tests": len(idx),
                    "elapsed_sec": time.time() - started, "per_test": [], "error": repr(exc),
                    "selected_indices": idx}
        if not outp.exists():
            status = "sandbox_violation" if proc.returncode else "runtime_error"
            err = (stderr or stdout or "worker produced no result")[-2000:]
            if proc.returncode in (-signal.SIGXCPU, -signal.SIGKILL): status = "timeout"
            return {"status": status, "passed_tests": 0, "total_tests": len(idx),
                    "elapsed_sec": time.time() - started, "per_test": [], "error": err,
                    "selected_indices": idx, "returncode": proc.returncode}
        try:
            result = json.loads(outp.read_text(encoding="utf-8"))
        except Exception as exc:
            result = {"status": "runtime_error", "passed_tests": 0, "total_tests": len(idx),
                      "per_test": [], "error": f"bad worker result: {exc}"}
        result["elapsed_sec"] = time.time() - started
        result["selected_indices"] = idx
        return result


def verify_response(text: str, gold: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
    code = extract_code(text)
    if not code:
        return {"status": "no_code", "passed_tests": 0, "total_tests": len(gold.get("inputs", [])),
                "elapsed_sec": 0.0, "per_test": []}
    result = verify_code(code, gold, **kwargs)
    result["extracted_code"] = code
    return result

# ------------------------- isolated worker below -------------------------
class _TestTimeout(Exception): pass
class _SandboxViolation(Exception): pass

def _alarm(_sig, _frame): raise _TestTimeout("per-test timeout")

def _decode_apps_arg(x: Any) -> Any:
    # codeparrot/APPS call tests commonly retain JSON literals for string args.
    if isinstance(x, str):
        s = x.strip()
        if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
            try: return json.loads(s)
            except Exception: return x
    return x

def _canon_call(x: Any) -> Any:
    if isinstance(x, tuple): return [_canon_call(v) for v in x]
    if isinstance(x, list): return [_canon_call(v) for v in x]
    if isinstance(x, dict): return {str(k): _canon_call(v) for k, v in x.items()}
    return x

def _equal_call_exact(a: Any, b: Any) -> bool:
    a, b = _canon_call(a), _canon_call(b)
    if isinstance(a, float) or isinstance(b, float):
        try: return math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=1e-6)
        except Exception: return False
    if type(a) != type(b): return False
    if isinstance(a, list): return len(a) == len(b) and all(_equal_call_exact(x, y) for x, y in zip(a, b))
    if isinstance(a, dict): return a.keys() == b.keys() and all(_equal_call_exact(a[k], b[k]) for k in a)
    return a == b

def _equal_call(a: Any, b: Any) -> bool:
    # Official APPS harness accepts both expected and expected[0] because some
    # serialized call-based outputs contain an extra singleton wrapper.
    if _equal_call_exact(a, b): return True
    return isinstance(b, list) and len(b) == 1 and _equal_call_exact(a, b[0])

def _equal_stdio(actual: str, expected: Any) -> bool:
    if isinstance(expected, list): expected = "\n".join(str(x) for x in expected)
    # Contest output semantics: whitespace separates tokens; order and token values remain significant.
    aa, bb = str(actual).split(), str(expected).split()
    if len(aa) != len(bb): return False
    for x, y in zip(aa, bb):
        if x == y: continue
        try:
            if not math.isclose(float(x), float(y), rel_tol=1e-6, abs_tol=1e-6): return False
        except Exception: return False
    return True

def _guard() -> None:
    import builtins, socket
    os.environ.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
    def blocked(*_a, **_k): raise _SandboxViolation("operation disabled by verifier")
    socket.socket = blocked
    socket.create_connection = blocked
    for name in ("system", "popen", "fork", "forkpty", "kill", "killpg", "open", "remove", "unlink",
                 "rename", "replace", "rmdir", "removedirs", "chdir", "chmod", "chown"):
        if hasattr(os, name): setattr(os, name, blocked)
    try:
        import subprocess as sp
        sp.Popen = sp.call = sp.run = sp.check_call = sp.check_output = blocked
    except Exception: pass
    # Candidate datasets do not require filesystem IO. Imports are preloaded by the execution prefix.
    builtins.open = blocked


def _namespace(code: str) -> Dict[str, Any]:
    prefix = """from typing import *
from collections import *
from functools import *
from itertools import *
from heapq import *
from bisect import *
from math import *
from statistics import *
from operator import *
import collections, functools, itertools, heapq, bisect, math, statistics, operator
import sys, json, re, string, random
sys.setrecursionlimit(600000)
"""
    ns: Dict[str, Any] = {"__name__": "__candidate__"}
    compiled = compile(prefix + "\n" + code, "<candidate>", "exec")
    _guard()
    exec(compiled, ns, ns)
    return ns

def _run_stdio(code: str, inp: Any) -> str:
    import builtins, contextlib, io
    if isinstance(inp, list): inp = "\n".join(str(x) for x in inp)
    stdin, stdout = io.StringIO(str(inp)), io.StringIO()
    old_input, old_open = builtins.input, builtins.open
    builtins.input = lambda prompt=None: stdin.readline().rstrip("\n")
    builtins.open = lambda file, mode="r", *a, **k: stdin if str(file) in ("/dev/stdin", "0") and "r" in mode else (_ for _ in ()).throw(_SandboxViolation("file access disabled"))
    old_stdin, old_stdout = sys.stdin, sys.stdout
    sys.stdin, sys.stdout = stdin, stdout
    try:
        ns = _namespace(code)
        # __name__ is intentionally not __main__; explicitly invoke common solve/main entrypoints if top-level emitted nothing.
        if not stdout.getvalue():
            fn = ns.get("solve") or ns.get("main")
            if callable(fn): fn()
    finally:
        sys.stdin, sys.stdout = old_stdin, old_stdout
        builtins.input, builtins.open = old_input, old_open
    return stdout.getvalue()

def _run_worker(req_path: str, out_path: str, timeout: float) -> int:
    req = json.loads(Path(req_path).read_text(encoding="utf-8")); code=req["code"]; gold=req["gold"]
    signal.signal(signal.SIGALRM, _alarm)
    tests=[]; passed=0; overall="pass"
    for j,(inp,exp) in enumerate(zip(gold["inputs"],gold["outputs"])):
        t=time.time(); status="pass"; detail={}
        try:
            signal.setitimer(signal.ITIMER_REAL, timeout)
            if gold.get("kind") == "call":
                ns=_namespace(code); obj=ns.get("Solution")() if isinstance(ns.get("Solution"),type) else ns
                fn=getattr(obj,gold["fn_name"]) if obj is not ns else ns.get(gold["fn_name"])
                if not callable(fn): raise AttributeError(f"missing callable {gold.get('fn_name')}")
                args=inp if isinstance(inp,list) else [inp]; args=[_decode_apps_arg(x) for x in args]
                actual=fn(*args); ok=_equal_call(actual,exp)
                if not ok: detail={"actual":repr(actual)[:1000],"expected":repr(exp)[:1000]}
            else:
                actual=_run_stdio(code,inp); ok=_equal_stdio(actual,exp)
                if not ok: detail={"actual":actual[:1000],"expected":str(exp)[:1000]}
            if not ok: status="wrong_answer"
        except SyntaxError as exc: status="compile_error";detail={"error":repr(exc)}
        except _TestTimeout as exc: status="timeout";detail={"error":repr(exc)}
        except _SandboxViolation as exc: status="sandbox_violation";detail={"error":repr(exc)}
        except BaseException as exc: status="runtime_error";detail={"error":repr(exc)[:1500]}
        finally: signal.setitimer(signal.ITIMER_REAL,0)
        passed += status=="pass"
        tests.append({"index":gold.get("original_indices",list(range(len(gold["inputs"]))))[j],"status":status,"elapsed_sec":time.time()-t,**detail})
        if status != "pass": overall=status; break
    result={"status":overall,"passed_tests":passed,"total_tests":len(gold["inputs"]),"per_test":tests}
    # Parent needs result creation despite builtins.open being disabled by candidate guard.
    payload=json.dumps(result,ensure_ascii=False).encode("utf-8")
    fd=_SAFE_OS_OPEN(out_path, os.O_WRONLY|os.O_CREAT|os.O_TRUNC, 0o600)
    try: _SAFE_OS_WRITE(fd,payload)
    finally: _SAFE_OS_CLOSE(fd)
    return 0

def main() -> None:
    ap=argparse.ArgumentParser();ap.add_argument("--worker",nargs=2,metavar=("REQUEST","RESULT"));ap.add_argument("--timeout",type=float,default=10);ap.add_argument("--response");ap.add_argument("--gold")
    a=ap.parse_args()
    if a.worker: raise SystemExit(_run_worker(a.worker[0],a.worker[1],a.timeout))
    if not a.gold: ap.error("--gold is required")
    gold=json.loads(Path(a.gold).read_text()); text=Path(a.response).read_text() if a.response else sys.stdin.read()
    print(json.dumps(verify_response(text,gold,timeout=a.timeout),ensure_ascii=False,indent=2))
if __name__ == "__main__": main()
