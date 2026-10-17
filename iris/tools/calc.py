"""Exact arithmetic, so answers don't depend on the model's mental maths."""

from __future__ import annotations

import ast
import math
import operator as op

from . import Tool

_OPS = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv, ast.FloorDiv: op.floordiv,
        ast.Mod: op.mod, ast.Pow: op.pow, ast.USub: op.neg, ast.UAdd: op.pos}
_FUNCS = {k: getattr(math, k) for k in ("sqrt", "sin", "cos", "tan", "log", "log10", "exp", "floor", "ceil",
                                        "factorial", "radians", "degrees")}
_FUNCS.update({"abs": abs, "round": round, "min": min, "max": max})
_NAMES = {"pi": math.pi, "e": math.e}


def evaluate(expr: str) -> float:
    expr = (expr.replace("×", "*").replace("÷", "/").replace("^", "**").replace(",", "")
            .replace("percent of", "/100*").replace("% of", "/100*").replace("%", "/100"))

    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            a, b = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Pow) and abs(b) > 1000:
                raise ValueError("exponent too large")
            return _OPS[type(n.op)](a, b)
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.operand))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _FUNCS:
            return _FUNCS[n.func.id](*[ev(a) for a in n.args])
        if isinstance(n, ast.Name) and n.id in _NAMES:
            return _NAMES[n.id]
        raise ValueError(f"not allowed: {ast.dump(n)[:60]}")

    return ev(ast.parse(expr.strip(), mode="eval"))


def run(args: dict) -> str:
    expr = str(args.get("expression") or "").strip()
    v = evaluate(expr)
    if isinstance(v, float) and v.is_integer() and abs(v) < 1e15:
        v = int(v)
    elif isinstance(v, float):
        v = round(v, 10)
    return f"{expr} = {v}"


TOOL = Tool(
    name="calculator",
    description="Exact arithmetic. Use it for any calculation instead of working it out yourself. "
                "Supports + - * / ** % sqrt() and parentheses, e.g. '17*6', '15% of 240', 'sqrt(2)'.",
    parameters={"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]},
    run=run, text_form="<<calc: EXPRESSION>>", text_arg="expression")
