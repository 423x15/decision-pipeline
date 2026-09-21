# agent/tools/calculator.py
from __future__ import annotations
import re
import ast
import operator

_SAFE_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos,
}

def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"Opération non supportée: {ast.dump(node)}")

def _extract_expression(query: str) -> str | None:
    matches = re.findall(r"[\d\s\+\-\*\/\(\)\.\^]+", query)
    for m in sorted(matches, key=len, reverse=True):
        candidate = m.strip()
        if len(candidate) >= 3 and any(c in candidate for c in "+-*/"):
            return candidate
    return None

def run(query: str, history: list[dict]) -> str:
    expr = _extract_expression(query)
    if not expr:
        return "Aucune expression mathématique trouvée dans la requête."
    try:
        tree = ast.parse(expr.replace("^", "**"), mode="eval")
        result = _safe_eval(tree.body)
        if result == int(result):
            return f"{expr.strip()} = {int(result)}"
        return f"{expr.strip()} = {result:.6g}"
    except Exception as exc:
        return f"Erreur d'évaluation de '{expr}': {exc}"
