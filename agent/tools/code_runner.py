# agent/tools/code_runner.py
from __future__ import annotations
import ast
import io
import sys
import textwrap
import traceback

_FORBIDDEN = {"open", "exec", "eval", "__import__", "compile",
              "breakpoint", "input", "exit", "quit"}

_FORBIDDEN_IMPORTS = {"os", "sys", "subprocess", "socket", "shutil",
                      "pathlib", "importlib", "ctypes", "builtins"}


def _safe_check(source: str) -> str | None:
    """Return an error string if the code uses forbidden constructs, else None."""
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return f"Syntax error: {e}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.id if isinstance(func, ast.Name) else None
            if name and name in _FORBIDDEN:
                return f"Forbidden call: {name}()"
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module.split(".")[0]]
            for n in names:
                if n in _FORBIDDEN_IMPORTS:
                    return f"Forbidden import: {n}"
    return None


def _extract_code(query: str) -> str:
    """Pull a ```python ... ``` block or return the query as-is."""
    if "```python" in query:
        start = query.index("```python") + 9
        end = query.index("```", start) if "```" in query[start:] else len(query)
        return query[start:end].strip()
    if "```" in query:
        start = query.index("```") + 3
        end = query.index("```", start) if "```" in query[start:] else len(query)
        return query[start:end].strip()
    return query.strip()


def run(query: str, history: list[dict]) -> str:
    code = _extract_code(query)
    code = textwrap.dedent(code)

    err = _safe_check(code)
    if err:
        return f"[Code Runner] Refused: {err}"

    stdout_buf = io.StringIO()
    local_ns: dict = {}
    try:
        old_stdout = sys.stdout
        sys.stdout = stdout_buf
        exec(compile(code, "<sandbox>", "exec"), {"__builtins__": __builtins__}, local_ns)  # noqa: S102
        sys.stdout = old_stdout
    except Exception:
        sys.stdout = old_stdout
        tb = traceback.format_exc(limit=3)
        return f"[Code Runner] Runtime error:\n{tb}"

    output = stdout_buf.getvalue().strip()

    # collect non-dunder locals as the result
    result_vars = {k: v for k, v in local_ns.items() if not k.startswith("_")}
    parts = []
    if output:
        parts.append(f"stdout:\n{output}")
    if result_vars:
        summary = ", ".join(f"{k}={repr(v)}" for k, v in list(result_vars.items())[:6])
        parts.append(f"variables: {summary}")
    return "[Code Runner] " + (" | ".join(parts) if parts else "executed (no output)")
