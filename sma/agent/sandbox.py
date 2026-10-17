"""Restricted execution of LLM-written Python against the pose-level robot API.

Failures are never swallowed: `try` is rejected, so every robot error reaches the harness
and becomes structured feedback. Every API call is traced with its arguments and result.
"""
import ast
import json
import math
import types

from ..env.robot_env import RobotError

FORBIDDEN_NODES = (ast.Import, ast.ImportFrom, ast.Try, ast.With, ast.AsyncFunctionDef, ast.Await,
                   ast.Global, ast.Nonlocal, ast.ClassDef, ast.Lambda)
FORBIDDEN_NAMES = {"eval", "exec", "compile", "open", "globals", "locals", "vars", "getattr",
                   "setattr", "delattr", "__import__", "input", "breakpoint", "help", "dir"}
SAFE_BUILTINS = {name: getattr(__builtins__, name) if hasattr(__builtins__, name) else __builtins__[name]
                 for name in ("abs", "min", "max", "round", "len", "range", "list", "dict", "tuple",
                              "float", "int", "str", "bool", "enumerate", "zip", "sorted", "sum",
                              "any", "all", "print", "isinstance", "ValueError", "RuntimeError",
                              "KeyError", "Exception", "reversed", "set")}
import os

# API level (see prompts.API_DOCS): "pose" = move_to + gripper, "task" = parameterized pick/place.
API_LEVEL = os.environ.get("SMA_API_LEVEL", "task")
API_NAMES = {"pose": ("observe", "move_to", "open_gripper", "close_gripper", "go_home", "wait"),
             "task": ("observe", "pick", "place", "go_home", "wait")}[API_LEVEL]


class ProgramRejected(ValueError):
    pass


def strip_code(text):
    text = text.strip()
    if "```" in text:
        blocks = text.split("```")
        candidates = [b[len("python"):] if b.startswith("python") else b for b in blocks[1::2]]
        text = max(candidates, key=len) if candidates else text
    return text.strip()


def check_source(source):
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise ProgramRejected(f"syntax_error: {exc.msg} at line {exc.lineno}") from exc
    for node in ast.walk(tree):
        if isinstance(node, FORBIDDEN_NODES):
            raise ProgramRejected(f"forbidden_statement: {type(node).__name__} at line "
                                  f"{getattr(node, 'lineno', '?')} (no imports, try/except, classes, lambdas)")
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            raise ProgramRejected(f"forbidden_name: {node.id}")
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise ProgramRejected(f"forbidden_attribute: {node.attr}")
    return tree


def _short(value, limit=600):
    text = json.dumps(value, default=str)
    return value if len(text) <= limit else text[:limit] + "..."


class Sandbox:
    def __init__(self, env, skills=None):
        self.env = env
        self.trace = []
        self.skills = skills or {}

    def _traced(self, name, fn):
        def call(*args, **kwargs):
            entry = {"call": name, "args": [round(a, 4) if isinstance(a, float) else a for a in args],
                     "kwargs": kwargs}
            self.trace.append(entry)
            try:
                result = fn(*args, **kwargs)
            except RobotError as exc:
                entry["error"] = {"reason": exc.reason, **exc.details}
                raise
            if name != "observe":
                entry["result"] = _short(result)
            return result
        return call

    def namespace(self):
        api = {name: self._traced(name, getattr(self.env, name)) for name in API_NAMES}
        ns = {"__builtins__": SAFE_BUILTINS, "math": math, **api}
        skill_fns = {}
        for skill_name, code in self.skills.items():
            skill_ns = {"__builtins__": SAFE_BUILTINS, "math": math, **api}
            exec(compile(check_source(code), f"<skill {skill_name}>", "exec"), skill_ns)
            fn = skill_ns.get(skill_name.split("@")[0])
            if fn is None:
                raise ProgramRejected(f"skill {skill_name} does not define its function")
            skill_fns[skill_name.split("@")[0]] = self._traced("skills." + skill_name, fn)
        ns["skills"] = types.SimpleNamespace(**skill_fns)
        return ns

    def run(self, source):
        """Returns (ok, error_dict or None). Robot state changes persist (online execution)."""
        try:
            tree = check_source(source)
        except ProgramRejected as exc:
            return False, {"type": "rejected", "message": str(exc)}
        ns = self.namespace()
        try:
            exec(compile(tree, "<program>", "exec"), ns)
        except RobotError as exc:
            return False, {"type": "robot_error", "reason": exc.reason, "details": exc.details}
        except Exception as exc:  # program bug: report type, message and line
            line = None
            tb = exc.__traceback__
            while tb is not None:
                if tb.tb_frame.f_code.co_filename in ("<program>",) or tb.tb_frame.f_code.co_filename.startswith("<skill"):
                    line = (tb.tb_frame.f_code.co_filename, tb.tb_lineno)
                tb = tb.tb_next
            return False, {"type": "program_error", "exception": type(exc).__name__,
                           "message": str(exc)[:400], "location": line}
        return True, None
