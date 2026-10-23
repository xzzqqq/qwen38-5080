"""实验请求（一次 submit = 一个 job）的 schema、argv 解析与提交期校验。

设计：
- 请求入队后不可修改，所以 commit、每个步骤的脚本、期望场景和数值参数都在
  submit 时解析好，写进 job 记录；执行器只按这份记录跑，不再猜。
- 只接受仓库里**已注册**的脚本路径（脚本 -> 允许的 kind）。新脚本要先在
  ``artifacts.py`` 里补结构化产物校验函数并注册，才允许提交；不能靠 LLM 读
  stdout 判断通过。
- ``argv`` 用数组保存、直接 ``subprocess`` 执行，不经 shell；``--out`` 和
  ``result`` 都必须落在本 job 的 ``{output}`` 内，避免复用旧结果或覆盖别的 job。
- 提交时就检查脚本在该 commit 里存在、``--only`` 的场景名合法、``--resume``
  等会破坏“本次测量”语义的 flag 不可用，把错误挡在 15 分钟的 GPU run 之前。
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

KINDS = ("check", "bench", "memory", "profile")
PLACEHOLDER = "{output}"
MAX_TIMEOUT_S = 86400.0
MAX_STEPS_PER_JOB = 32
_FULL_SHA = re.compile(r"^[0-9a-f]{40}$")


class RequestError(ValueError):
    """请求不合法：字段、commit、命令或产物路径。"""


@dataclass(frozen=True)
class ScriptSpec:
    """一个已注册的实验脚本：允许的 kind、输出目录 flag、场景表变量名。"""

    path: str
    kinds: tuple[str, ...]
    out_flag: str = "--out"
    scenarios_attr: str | None = None
    forbidden: tuple[str, ...] = ()


# 只注册有结构化产物校验的脚本。新增脚本 = 先补 artifacts.VALIDATORS 再进这里。
SCRIPTS: dict[str, ScriptSpec] = {
    "scripts/m3_check.py": ScriptSpec(
        "scripts/m3_check.py",
        ("check",),
        scenarios_attr="SCENARIOS",
        forbidden=("--worker",),
    ),
    "bench/e2e/decode.py": ScriptSpec(
        "bench/e2e/decode.py",
        ("bench",),
        scenarios_attr="SCENARIOS",
        # --resume 会复用旧场景结果，--list 不产出结果，--worker 是内部参数。
        forbidden=("--worker", "--resume", "--list"),
    ),
    "scripts/m3_memory.py": ScriptSpec(
        "scripts/m3_memory.py",
        ("memory",),
        scenarios_attr="SCENARIOS",
        forbidden=("--worker",),
    ),
    "scripts/m3_profile.sh": ScriptSpec(
        "scripts/m3_profile.sh",
        ("profile",),
        out_flag="--out",
    ),
}

# 参与校验的带值 flag；没列出的 flag 原样传给脚本，不参与校验。
VALUE_FLAGS = (
    "--out",
    "--only",
    "--steps",
    "--repeats",
    "--repeats-non-graph",
    "--chunk-size",
    "--graph-warmup",
    "--fixtures",
    "--smoke-max-new-tokens",
)
NUMERIC_FLAGS = {
    "--steps": 1,
    "--repeats": 1,
    "--repeats-non-graph": 1,
    "--chunk-size": 1,
    "--graph-warmup": 1,
    "--smoke-max-new-tokens": 1,
}

# 每个 kind 的脚本默认值（与脚本 argparse 的 default 保持一致）。
KIND_DEFAULTS: dict[str, dict[str, int]] = {
    "check": {"steps": 16, "smoke_max_new_tokens": 2048},
    "bench": {"steps": 511, "repeats": 5, "chunk_size": 128, "graph_warmup": 3},
    "memory": {"steps": 511, "chunk_size": 128},
    "profile": {"steps": 8},
}


# ------------------------------------------------------------------ argv 解析
def parse_flags(argv: list[str]) -> dict[str, list[str]]:
    """抽取校验用的 flag 值；重复的 flag 保留全部，由调用方决定是否报错。"""
    values: dict[str, list[str]] = {}
    for index, token in enumerate(argv):
        name, eq, inline = token.partition("=")
        if name not in VALUE_FLAGS:
            continue
        if eq:
            values.setdefault(name, []).append(inline)
            continue
        if index + 1 >= len(argv):
            raise RequestError(f"{name} 缺少值")
        values.setdefault(name, []).append(argv[index + 1])
    return values


def _one(values: dict[str, list[str]], flag: str) -> str | None:
    found = values.get(flag) or []
    if len(found) > 1:
        raise RequestError(f"{flag} 重复指定（{found}）")
    return found[0] if found else None


def _int_one(
    values: dict[str, list[str]], flag: str, default: int, minimum: int = 1
) -> int:
    raw = _one(values, flag)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise RequestError(f"{flag} 不是整数: {raw!r}") from exc
    if value < minimum:
        raise RequestError(f"{flag} 必须 >= {minimum}，得到 {value}")
    return value


def _only_names(
    values: dict[str, list[str]], known: list[str] | None
) -> tuple[str, ...] | None:
    """返回显式 ``--only`` 的场景名；``all``/未给返回 None。"""
    raw = _one(values, "--only")
    if raw is None or raw.strip() in ("", "all"):
        return None
    names = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not names:
        raise RequestError("--only 为空")
    if len(set(names)) != len(names):
        raise RequestError(f"--only 有重复场景: {list(names)}")
    if known is not None:
        unknown = [name for name in names if name not in known]
        if unknown:
            raise RequestError(
                f"--only 有未知场景 {unknown}；可用: {', '.join(known)}"
            )
    return names


# ------------------------------------------------------------------ 路径校验
def _template_rel(template: str, what: str) -> PurePosixPath:
    """把 ``{output}/a/b`` 变成相对 ``a/b``，并拒绝跳出 output 的路径。"""
    if not template.startswith(PLACEHOLDER + "/"):
        raise RequestError(f"{what} 必须以 {PLACEHOLDER}/ 开头: {template!r}")
    rel = PurePosixPath(template[len(PLACEHOLDER) + 1 :])
    if not rel.parts or ".." in rel.parts or rel.is_absolute():
        raise RequestError(f"{what} 不是 output 内的合法相对路径: {template!r}")
    return rel


def _expand_template(template: str, output_dir: Path) -> str:
    return template.replace(PLACEHOLDER, str(output_dir))


def _script_relative(token: str, root: Path) -> str | None:
    if token.startswith("-") or PLACEHOLDER in token:
        return None
    path = Path(token)
    if path.is_absolute():
        try:
            return path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            return None
    rel = PurePosixPath(token)
    if rel.parts and rel.parts[0] == "..":
        return None
    return str(rel)


def find_script(argv: tuple[str, ...] | list[str], root: Path) -> str | None:
    hits: list[str] = []
    for token in argv:
        rel = _script_relative(token, root)
        if rel is not None and rel in SCRIPTS:
            hits.append(rel)
    if len(hits) > 1:
        raise RequestError(f"一个步骤里出现多个已注册脚本: {sorted(set(hits))}")
    return hits[0] if hits else None


def scenario_names_from_source(source: str, attr: str | None) -> list[str] | None:
    """从脚本源码的字面量场景表里取场景名；解析不出来返回 None（不做场景校验）。"""
    if not attr:
        return None
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == attr for t in node.targets):
            continue
        try:
            value = ast.literal_eval(node.value)
        except Exception:  # noqa: BLE001 - 动态构造的场景表就放弃校验
            return None
        if isinstance(value, dict):
            names = [str(key) for key in value]
        elif isinstance(value, (list, tuple)):
            names = [str(item) for item in value]
        else:
            return None
        return names or None
    return None


# ------------------------------------------------------------------ git 校验
def _git(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def resolve_commit(root: Path, commit: str) -> str:
    """校验完整 SHA 在本仓库存在，返回 40 位小写 SHA。"""
    if not isinstance(commit, str) or not _FULL_SHA.match(commit.lower()):
        raise RequestError("commit 必须是完整的 40 位 commit SHA")
    proc = _git(root, "rev-parse", "--verify", "--quiet", f"{commit}^{{commit}}")
    if proc.returncode != 0:
        raise RequestError(
            f"commit 不存在或不是 commit: {commit}（{proc.stderr.strip()[:200]}）"
        )
    sha = proc.stdout.strip().lower()
    if not _FULL_SHA.match(sha):
        raise RequestError(f"commit 解析失败: {commit} -> {sha!r}")
    return sha


def commit_has_file(root: Path, commit: str, rel: str) -> bool:
    return _git(root, "cat-file", "-e", f"{commit}:{rel}").returncode == 0


def commit_file_text(root: Path, commit: str, rel: str) -> str | None:
    proc = _git(root, "show", f"{commit}:{rel}")
    if proc.returncode != 0:
        return None
    return proc.stdout


# ------------------------------------------------------------------ 数据结构
@dataclass(frozen=True)
class ResolvedStep:
    index: int
    kind: str
    argv: tuple[str, ...]
    result: str
    script: str
    flags: dict[str, str]
    scenarios: tuple[str, ...] | None

    def expand_argv(self, output_dir: Path) -> list[str]:
        return [_expand_template(token, output_dir) for token in self.argv]

    def result_path(self, output_dir: Path) -> Path:
        return Path(output_dir) / str(_template_rel(self.result, "result"))


@dataclass(frozen=True)
class Request:
    raw: dict[str, Any]
    session_id: str
    commit_requested: str
    commit: str
    timeout_s: float
    steps: tuple[ResolvedStep, ...]
    sha256: str


@dataclass(frozen=True)
class Expectation:
    """一个步骤执行时的期望值；校验器用它判断产物是否属于本次请求。"""

    kind: str
    argv: tuple[str, ...]
    commit: str
    output_dir: Path
    result_path: Path
    worktree: Path | None
    scenarios: tuple[str, ...] | None
    steps: int
    repeats: int | None
    repeats_non_graph: int | None
    chunk_size: int
    graph_warmup: int
    fixtures: str | None
    smoke_max_new_tokens: int | None

    @classmethod
    def build(
        cls,
        step: ResolvedStep,
        request: Request,
        output_dir: Path,
        worktree: Path | None = None,
    ) -> "Expectation":
        defaults = KIND_DEFAULTS[step.kind]
        values = {name: [value] for name, value in step.flags.items()}
        repeats = _int_one(values, "--repeats", defaults["repeats"]) if "repeats" in defaults else None
        repeats_non_graph = (
            _int_one(values, "--repeats-non-graph", 0, minimum=1)
            if _one(values, "--repeats-non-graph") is not None
            else None
        )
        return cls(
            kind=step.kind,
            argv=step.argv,
            commit=request.commit,
            output_dir=Path(output_dir),
            result_path=step.result_path(output_dir),
            worktree=worktree,
            scenarios=step.scenarios,
            steps=_int_one(values, "--steps", defaults["steps"]),
            repeats=repeats,
            repeats_non_graph=repeats_non_graph,
            chunk_size=_int_one(values, "--chunk-size", defaults.get("chunk_size", 128)),
            graph_warmup=_int_one(values, "--graph-warmup", defaults.get("graph_warmup", 3)),
            fixtures=_one(values, "--fixtures"),
            smoke_max_new_tokens=(
                _int_one(
                    values,
                    "--smoke-max-new-tokens",
                    defaults.get("smoke_max_new_tokens", 2048),
                )
                if "smoke_max_new_tokens" in defaults
                else None
            ),
        )

    def expected_repeats(self, execution: str) -> int | None:
        if self.repeats is None:
            return None
        if execution != "graph" and self.repeats_non_graph is not None:
            return self.repeats_non_graph
        return self.repeats


# ------------------------------------------------------------------ 请求校验
def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RequestError(message)


def _check_keys(obj: dict[str, Any], allowed: set[str], where: str) -> None:
    unknown = sorted(set(obj) - allowed)
    missing = sorted(allowed - set(obj))
    if unknown:
        raise RequestError(f"{where} 有未知字段: {unknown}")
    if missing:
        raise RequestError(f"{where} 缺字段: {missing}")


def parse_request_object(obj: Any, root: Path) -> Request:
    """校验请求对象；提交和入队后重放都走这里。"""
    _require(isinstance(obj, dict), "请求必须是 JSON 对象")
    assert isinstance(obj, dict)
    _check_keys(obj, {"session_id", "commit", "timeout_s", "steps"}, "请求")

    session_id = obj["session_id"]
    _require(isinstance(session_id, str) and bool(session_id.strip()), "session_id 不能为空")
    assert isinstance(session_id, str)
    session_id = session_id.strip()
    _require(len(session_id) <= 200, "session_id 太长（>200）")
    _require(
        not any(ch in session_id for ch in "\x00\r\n\t"),
        "session_id 不能包含控制字符",
    )

    commit_requested = obj["commit"]
    _require(isinstance(commit_requested, str), "commit 必须是字符串")
    assert isinstance(commit_requested, str)
    commit = resolve_commit(root, commit_requested)

    timeout_s = obj["timeout_s"]
    _require(
        isinstance(timeout_s, (int, float)) and not isinstance(timeout_s, bool),
        "timeout_s 必须是数字",
    )
    timeout_s = float(timeout_s)
    _require(math.isfinite(timeout_s) and timeout_s > 0, "timeout_s 必须为正数")
    _require(timeout_s <= MAX_TIMEOUT_S, f"timeout_s 不能超过 {MAX_TIMEOUT_S:.0f} 秒")

    steps_obj = obj["steps"]
    _require(isinstance(steps_obj, list) and bool(steps_obj), "steps 必须是非空数组")
    assert isinstance(steps_obj, list)
    _require(
        len(steps_obj) <= MAX_STEPS_PER_JOB,
        f"steps 最多 {MAX_STEPS_PER_JOB} 个",
    )

    steps: list[ResolvedStep] = []
    result_templates: set[str] = set()
    for index, item in enumerate(steps_obj):
        where = f"steps[{index}]"
        _require(isinstance(item, dict), f"{where} 必须是对象")
        assert isinstance(item, dict)
        _check_keys(item, {"kind", "argv", "result"}, where)
        kind = item["kind"]
        _require(kind in KINDS, f"{where}.kind 必须是 {KINDS} 之一")
        argv = item["argv"]
        _require(isinstance(argv, list) and bool(argv), f"{where}.argv 必须是非空数组")
        assert isinstance(argv, list)
        for pos, token in enumerate(argv):
            _require(
                isinstance(token, str) and token != "", f"{where}.argv[{pos}] 必须是非空字符串"
            )
            _require("\x00" not in token, f"{where}.argv[{pos}] 不能含 NUL")
        argv = tuple(argv)  # type: ignore[assignment]

        result = item["result"]
        _require(isinstance(result, str), f"{where}.result 必须是字符串")
        assert isinstance(result, str)
        result_rel = _template_rel(result, f"{where}.result")
        _require(result_rel.suffix == ".json", f"{where}.result 必须是 .json 文件")
        _require(result not in result_templates, f"{where}.result 与其他步骤重复")
        result_templates.add(result)

        script = find_script(argv, root)
        _require(
            script is not None,
            f"{where} 没有已注册的实验脚本；已注册: {sorted(SCRIPTS)}"
            "（新脚本必须先补结构化产物校验）",
        )
        assert script is not None
        spec = SCRIPTS[script]
        _require(
            kind in spec.kinds,
            f"{where}.kind={kind} 与脚本 {script} 不匹配（允许 {spec.kinds}）",
        )
        _require(
            commit_has_file(root, commit, script),
            f"{where}: commit {commit[:8]} 里没有 {script}",
        )
        for forbidden in spec.forbidden:
            _require(
                forbidden not in argv,
                f"{where}: 不允许 {forbidden}（会复用旧结果或跳过测量）",
            )

        values = parse_flags(list(argv))
        out_values = values.get(spec.out_flag) or []
        _require(bool(out_values), f"{where}: 缺少 {spec.out_flag}（必须是 {PLACEHOLDER} 下的目录）")
        _require(len(out_values) == 1, f"{where}: {spec.out_flag} 重复指定")
        out_template = out_values[0]
        if out_template == PLACEHOLDER:
            out_rel = PurePosixPath(".")
        else:
            out_rel = _template_rel(out_template, f"{where} 的 {spec.out_flag}")
        inside = out_rel == PurePosixPath(".") or out_rel in result_rel.parents
        _require(
            inside,
            f"{where}: result {result!r} 不在 {spec.out_flag} {out_template!r} 目录下",
        )

        flags: dict[str, str] = {}
        for flag in VALUE_FLAGS:
            value = _one(values, flag)
            if value is not None:
                flags[flag] = value
        for flag, minimum in NUMERIC_FLAGS.items():
            if flag in flags:
                _int_one({flag: [flags[flag]]}, flag, minimum, minimum)

        source = commit_file_text(root, commit, script)
        known = scenario_names_from_source(source or "", spec.scenarios_attr)
        explicit = _only_names(values, known)
        if explicit is not None:
            scenarios: tuple[str, ...] | None = explicit
        elif known is not None:
            scenarios = tuple(known)
        else:
            scenarios = None

        steps.append(
            ResolvedStep(
                index=index,
                kind=kind,
                argv=argv,
                result=result,
                script=script,
                flags=flags,
                scenarios=scenarios,
            )
        )

    raw = obj
    sha = hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return Request(
        raw=raw,
        session_id=session_id,
        commit_requested=commit_requested,
        commit=commit,
        timeout_s=timeout_s,
        steps=tuple(steps),
        sha256=sha,
    )


def load_request(path: Path, root: Path) -> Request:
    try:
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RequestError(f"请求文件不存在: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RequestError(f"请求不是合法 JSON: {exc}") from exc
    return parse_request_object(obj, root)
