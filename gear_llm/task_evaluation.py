import ast
import json
import re
import subprocess
import sys
import tempfile
import textwrap
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

import torch

from gear_llm.adaptive_generator import (
    AdaptiveGenerationConfig,
    load_adaptive_models,
)
from gear_llm.hybrid_router import classify_prompt, choose_mode, generate_with_mode
from gear_llm.model_loader import (
    get_cheap_tokenizer,
    get_expensive_tokenizer,
    get_model_runtime_metadata,
    prompt_format_metadata,
)
from gear_llm.quality_benchmark import (
    estimated_saved_percent,
    generate_greedy_with_model,
)
from gear_llm.prompt_router import (
    classify_prompt_router_ml_v1,
    classify_prompt_router_v1,
    classify_prompt_router_v2,
    load_prompt_router_ml_model,
)
from gear_llm.report import save_csv
from gear_llm.runtime_profiler import (
    COUNT_FIELDS,
    TIME_FIELDS,
    RuntimeProfiler,
    maybe_profiler,
)


TASK_MODES = (
    "expensive_only",
    "cheap_only",
    "adaptive_calibrated",
    "adaptive_guarded_v3",
    "adaptive_code_quality",
    "speculative_adaptive",
    "prompt_router_v1",
    "prompt_router_v2",
    "prompt_router_ml_v1",
    "hybrid",
)
DIFFICULTIES = {"easy", "medium", "hard"}
MATH_ANSWER_TYPES = {"number", "expression", "boolean", "choice"}
LOGIC_ANSWER_TYPES = {"label", "choice"}
DERIVED_PROFILE_FIELDS = (
    "average_time_per_generated_token",
    "average_cheap_forward_time",
    "average_expensive_forward_time",
    "routing_overhead_time_seconds",
)
PROFILE_FIELDS = TIME_FIELDS + COUNT_FIELDS + DERIVED_PROFILE_FIELDS


def load_eval_tasks(path: str | Path) -> list[dict]:
    tasks = []
    dataset_path = Path(path)

    with dataset_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            item = json.loads(stripped)
            missing = {"id", "category", "prompt", "difficulty"} - set(item)
            if missing:
                fields = ", ".join(sorted(missing))
                raise ValueError(
                    f"Line {line_number} of {dataset_path} missing fields: {fields}"
                )
            if item["difficulty"] not in DIFFICULTIES:
                raise ValueError(
                    f"Task {item['id']} has invalid difficulty: {item['difficulty']}"
                )

            category = item["category"]
            if category == "math":
                missing_math = {
                    "expected_answer",
                    "acceptable_answers",
                    "answer_type",
                } - set(item)
                if missing_math:
                    fields = ", ".join(sorted(missing_math))
                    raise ValueError(f"Task {item['id']} missing fields: {fields}")
                if item["answer_type"] not in MATH_ANSWER_TYPES:
                    raise ValueError(
                        f"Task {item['id']} has invalid answer_type: "
                        f"{item['answer_type']}"
                    )
            elif category == "logic":
                answer_type = item.get("answer_type", "label")
                if answer_type not in LOGIC_ANSWER_TYPES:
                    raise ValueError(
                        f"Task {item['id']} has invalid answer_type: {answer_type}"
                    )
                if answer_type == "choice":
                    missing_logic = {"expected_answer", "acceptable_answers"} - set(item)
                else:
                    missing_logic = {"expected_label", "acceptable_labels"} - set(item)
                if missing_logic:
                    fields = ", ".join(sorted(missing_logic))
                    raise ValueError(f"Task {item['id']} missing fields: {fields}")
            elif category == "code":
                missing_code = {"function_name", "tests"} - set(item)
                if missing_code:
                    fields = ", ".join(sorted(missing_code))
                    raise ValueError(f"Task {item['id']} missing fields: {fields}")
            else:
                raise ValueError(f"Unsupported category in task {item['id']}: {category}")

            tasks.append(item)

    return tasks


def _parse_filter_values(values: list[str] | str | None) -> list[str] | None:
    if values is None:
        return None
    if isinstance(values, str):
        parsed = [value.strip() for value in values.split(",")]
    else:
        parsed = [str(value).strip() for value in values]
    parsed = [value for value in parsed if value]
    return parsed or None


def filter_eval_tasks(
    tasks: list[dict],
    categories: list[str] | str | None = None,
    difficulties: list[str] | str | None = None,
    limit: int | None = None,
) -> list[dict]:
    category_filter = set(_parse_filter_values(categories) or [])
    difficulty_filter = set(_parse_filter_values(difficulties) or [])

    invalid_difficulties = difficulty_filter - DIFFICULTIES
    if invalid_difficulties:
        values = ", ".join(sorted(invalid_difficulties))
        raise ValueError(f"Invalid difficulties: {values}")

    filtered = []
    for task in tasks:
        if category_filter and task["category"] not in category_filter:
            continue
        if difficulty_filter and task["difficulty"] not in difficulty_filter:
            continue
        filtered.append(task)

    if limit is not None:
        filtered = filtered[: max(0, limit)]

    return filtered


def resolve_task_modes(modes: list[str] | str | None = None) -> list[str]:
    requested = _parse_filter_values(modes)
    if requested is None:
        return list(TASK_MODES)

    invalid = [mode for mode in requested if mode not in TASK_MODES]
    if invalid:
        values = ", ".join(invalid)
        valid = ", ".join(TASK_MODES)
        raise ValueError(f"Invalid task evaluation modes: {values}. Valid modes: {valid}")

    return requested


def _strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


def normalize_math_text(text: str) -> str:
    text = _strip_accents(str(text).lower())
    text = re.sub(
        r"```(?:\w+)?\s*(.*?)```",
        r"\1",
        text,
        flags=re.DOTALL,
    )
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"[*_#$]", "", text)
    text = text.replace("\\left", "").replace("\\right", "")
    text = re.sub(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"(\1)/\2", text)
    text = text.replace("\\cdot", "*").replace("\\times", "*")
    text = text.replace("−", "-").replace("–", "-")
    text = text.replace("\\", "")
    text = re.sub(r"(?<=\d),(?=\d)", ".", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _compact_math(text: str) -> str:
    return re.sub(r"\s+", "", normalize_math_text(text))


def _strip_assignment(text: str) -> str:
    return re.sub(
        r"^(?:[a-z]|[a-z]\([a-z]\)|f\^-?1\([a-z]\)|g\^-?1\([a-z]\)|"
        r"h\^-?1\([a-z]\)|answer|result|resposta)=",
        "",
        text,
    )


def _clean_extracted_math_answer(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"^[`*_=\s:]+", "", text)
    text = re.sub(r"[`*_]+$", "", text).strip()
    text = re.sub(r"(?:</s>|<\|endoftext\|>)$", "", text).strip()

    while text and text[-1] in ".。!;":
        text = text[:-1].rstrip()

    return text


def _preferred_math_answer_texts(generated_text: str) -> list[str]:
    markers = (
        r"Final\s+Answer\s*:",
        r"Answer\s*:",
        r"The\s+answer\s+is",
    )
    candidates = []
    for marker in markers:
        pattern = rf"{marker}\s*([^\n\r]+)"
        for match in re.finditer(pattern, str(generated_text), flags=re.IGNORECASE):
            candidate = _clean_extracted_math_answer(match.group(1))
            if candidate:
                candidates.append(candidate)

    return candidates


def _number_match(generated: str, answer: str) -> bool:
    answer_core = _strip_assignment(_compact_math(answer))
    if not answer_core:
        return False

    normalized_generated = normalize_math_text(generated)
    compact_generated = _compact_math(generated)
    escaped = re.escape(answer_core)
    patterns = (
        rf"(?<![a-z0-9.])(?:[a-z]|answer|result|resposta)?\s*=\s*{escaped}(?![a-z0-9.])",
        rf"(?<![a-z0-9.]){escaped}(?![a-z0-9.])",
    )

    for pattern in patterns:
        if re.search(pattern, normalized_generated):
            return True
        if re.search(pattern.replace(r"\s*", ""), compact_generated):
            return True

    return False


def _expression_match(generated: str, answer: str) -> bool:
    generated_compact = _compact_math(generated)
    answer_compact = _compact_math(answer)
    answer_core = _strip_assignment(answer_compact)

    candidates = {answer_compact, answer_core}
    candidates.update(
        {
            f"x={answer_core}",
            f"y={answer_core}",
            f"answer={answer_core}",
            f"result={answer_core}",
            f"f^-1(x)={answer_core}",
            f"g^-1(x)={answer_core}",
            f"h^-1(x)={answer_core}",
        }
    )

    return any(candidate and candidate in generated_compact for candidate in candidates)


def _math_answer_matches(text: str, answer: str, answer_type: str) -> bool:
    if answer_type == "number":
        return _number_match(text, answer)
    if answer_type in {"expression", "choice"}:
        return _expression_match(text, answer) or _number_match(text, answer)
    if answer_type == "boolean":
        return _expression_match(text, answer)
    return False


def evaluate_math(task: dict, generated_text: str) -> dict:
    expected_answers = [task["expected_answer"]]
    expected_answers.extend(task.get("acceptable_answers", []))
    answer_type = task.get("answer_type", "number")
    candidate_texts = _preferred_math_answer_texts(generated_text)
    candidate_texts.append(generated_text)

    for answer in expected_answers:
        matched = any(
            _math_answer_matches(candidate, str(answer), answer_type)
            for candidate in candidate_texts
        )

        if matched:
            return {
                "inferred_answer": str(answer),
                "inferred_label": "",
                "passed": True,
                "score": 1.0,
                "error": "",
                "failure_reason": "",
                "test_count": 0,
                "passed_tests": 0,
            }

    return {
        "inferred_answer": "",
        "inferred_label": "",
        "passed": False,
        "score": 0.0,
        "error": "",
        "failure_reason": "expected_answer_not_found",
        "test_count": 0,
        "passed_tests": 0,
    }


LOGIC_LABEL_PATTERNS = {
    "unknown": (
        r"\bunknown\b",
        r"\bunclear\b",
        r"\bnot enough\b",
        r"\binsufficient information\b",
        r"\binsufficient\b",
        r"\bcannot determine\b",
        r"\bcan't determine\b",
        r"\bindeterminate\b",
        r"\bnao e possivel determinar\b",
        r"\binformacao insuficiente\b",
        r"\bindeterminado\b",
        r"\bincerto\b",
    ),
    "depends": (
        r"\bdepends\b",
        r"\bit depends\b",
        r"\bdepending\b",
        r"\bdepende\b",
    ),
    "deny": (
        r"\bdeny\b",
        r"\bdenied\b",
        r"\bshould not\b",
        r"\bnot allowed\b",
        r"\bnot be allowed\b",
        r"\bno\b",
        r"\bblock\b",
        r"\bblocked\b",
        r"\breject\b",
        r"\brejected\b",
        r"\bprohibit\b",
        r"\bforbid\b",
        r"\bnegar\b",
        r"\bnegado\b",
        r"\bbloquear\b",
        r"\bbloqueado\b",
        r"\bnao\b",
        r"\bproibido\b",
    ),
    "allow": (
        r"\ballow\b",
        r"\ballowed\b",
        r"\byes\b",
        r"\bpermit\b",
        r"\bpermitted\b",
        r"\bgrant\b",
        r"\bsim\b",
        r"\bpermitir\b",
        r"\bpermitido\b",
        r"\bautorizar\b",
        r"\bautorizado\b",
    ),
}


def infer_logic_label(text: str) -> str:
    normalized = _strip_accents(str(text).lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    first_chunk = re.split(r"[.\n:;,-]", normalized, maxsplit=1)[0]

    starts = (
        ("unknown", (r"unknown", r"unclear", r"nao e possivel", r"incerto")),
        ("depends", (r"depends", r"it depends", r"depende")),
        ("deny", (r"deny", r"denied", r"no", r"not allowed", r"negar", r"negado", r"nao")),
        ("allow", (r"allow", r"allowed", r"yes", r"permit", r"sim", r"permitir")),
    )
    for label, patterns in starts:
        if any(re.match(rf"^\s*{pattern}\b", first_chunk) for pattern in patterns):
            return label

    for label in ("unknown", "depends", "deny", "allow"):
        for pattern in LOGIC_LABEL_PATTERNS[label]:
            if re.search(pattern, normalized):
                return label

    return "unknown"


def infer_logic_choice(text: str) -> str:
    cleaned = re.sub(r"```(?:\w+)?\s*(.*?)```", r"\1", str(text), flags=re.DOTALL)
    cleaned = re.sub(r"`([^`]*)`", r"\1", cleaned).strip()
    compact = re.sub(r"\s+", " ", cleaned)

    explicit_patterns = (
        r"\b(?:answer|option|letter|resposta|opcao|opção|letra)\s*"
        r"(?:is|é|:)?\s*[\(\[]?([A-Da-d])\b",
        r"\b(?:the correct answer is|a resposta correta e|a resposta correta é)\s*"
        r"[\(\[]?([A-Da-d])\b",
    )
    for pattern in explicit_patterns:
        match = re.search(pattern, compact, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()

    start_match = re.match(r"^\s*[\(\[]?([A-D])[\)\].,:;\s]", cleaned)
    if start_match:
        remainder = cleaned[start_match.end() :].lstrip().lower()
        if not remainder.startswith(("resposta", "answer")):
            return start_match.group(1).upper()

    single = re.fullmatch(r"\s*[\(\[]?([A-Da-d])[\)\].]?\s*", cleaned)
    if single:
        return single.group(1).upper()

    return ""


def evaluate_logic(task: dict, generated_text: str) -> dict:
    if task.get("answer_type") == "choice":
        inferred_answer = infer_logic_choice(generated_text)
        acceptable = {str(answer).upper() for answer in task.get("acceptable_answers", [])}
        passed = inferred_answer in acceptable

        return {
            "inferred_answer": inferred_answer,
            "inferred_label": "",
            "passed": passed,
            "score": 1.0 if passed else 0.0,
            "error": "",
            "failure_reason": "" if passed else f"inferred_{inferred_answer or 'none'}",
            "test_count": 0,
            "passed_tests": 0,
        }

    inferred_label = infer_logic_label(generated_text)
    acceptable = set(task.get("acceptable_labels", []))
    passed = inferred_label in acceptable

    return {
        "inferred_answer": "",
        "inferred_label": inferred_label,
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "error": "",
        "failure_reason": "" if passed else f"inferred_{inferred_label}",
        "test_count": 0,
        "passed_tests": 0,
    }


def _extract_fenced_code(text: str, function_name: str) -> str:
    blocks = re.findall(
        r"```(?:python|py)?\s*(.*?)```",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    for block in blocks:
        if re.search(rf"\bdef\s+{re.escape(function_name)}\s*\(", block):
            return block.strip()

    return blocks[0].strip() if blocks else ""


def _safe_import_prefix(lines: list[str], end: int) -> list[str]:
    imports = []
    for line in lines[:end]:
        stripped = line.strip()
        if re.match(r"^(import\s+math|from\s+math\s+import\s+[a-zA-Z0-9_,\s]+)$", stripped):
            imports.append(stripped)
    return imports


def _trim_function_block(code: str, function_name: str) -> str:
    lines = code.replace("\r\n", "\n").split("\n")
    start = None
    pattern = re.compile(
        rf"^\s*def\s+{re.escape(function_name)}\s*\([^)]*\)\s*(?:->\s*[^:]+)?\s*:"
    )

    for index, line in enumerate(lines):
        if pattern.search(line):
            start = index
            break

    if start is None:
        return ""

    block = _safe_import_prefix(lines, start)
    block.append(lines[start])
    for line in lines[start + 1 :]:
        stripped = line.strip()
        if not stripped:
            block.append(line)
            continue
        if line.startswith((" ", "\t")):
            block.append(line)
            continue
        break

    return "\n".join(block).strip()


def extract_python_code(generated_text: str, function_name: str) -> str:
    fenced = _extract_fenced_code(generated_text, function_name)
    if fenced:
        trimmed = _trim_function_block(fenced, function_name)
        return trimmed or fenced

    match = re.search(
        rf"\bdef\s+{re.escape(function_name)}\s*\(",
        generated_text,
    )
    if not match:
        return ""

    prefix_start = max(0, match.start() - 500)
    return _trim_function_block(generated_text[prefix_start:], function_name)


UNSAFE_CALL_NAMES = {
    "__import__",
    "compile",
    "eval",
    "exec",
    "getattr",
    "globals",
    "input",
    "locals",
    "open",
    "setattr",
}
SAFE_MATH_IMPORTS = {
    "acos",
    "asin",
    "atan",
    "ceil",
    "cos",
    "degrees",
    "e",
    "exp",
    "fabs",
    "floor",
    "gcd",
    "isclose",
    "log",
    "pi",
    "pow",
    "radians",
    "sin",
    "sqrt",
    "tan",
    "trunc",
}


def validate_code_safety(code: str) -> tuple[bool, str]:
    try:
        tree = ast.parse(code)
    except SyntaxError as error:
        return False, f"syntax_error: {error}"

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name != "math" for alias in node.names):
                return False, "only_math_import_is_allowed"
        if isinstance(node, ast.ImportFrom):
            if node.module != "math":
                return False, "only_from_math_import_is_allowed"
            for alias in node.names:
                if alias.name.startswith("_") or alias.name not in SAFE_MATH_IMPORTS:
                    return False, f"unsafe_math_import: {alias.name}"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in UNSAFE_CALL_NAMES:
                return False, f"unsafe_call: {node.func.id}"
        if isinstance(node, ast.Attribute) and node.attr in {
            "system",
            "popen",
            "remove",
            "unlink",
            "rmdir",
            "mkdir",
            "write",
            "connect",
            "request",
        }:
            return False, f"unsafe_attribute: {node.attr}"

    return True, ""


def validate_tests_safety(tests: list[dict]) -> tuple[bool, str]:
    for test in tests:
        raw_assert = test.get("assert") or test.get("raw_assert")
        if not raw_assert:
            continue

        try:
            tree = ast.parse(raw_assert)
        except SyntaxError as error:
            return False, f"test_syntax_error: {error}"

        if not tree.body or not isinstance(tree.body[0], ast.Assert):
            return False, "raw_test_must_be_assert_statement"

        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                return False, "imports_not_allowed_in_tests"
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in UNSAFE_CALL_NAMES:
                    return False, f"unsafe_test_call: {node.func.id}"
            if isinstance(node, ast.Attribute) and node.attr in {
                "system",
                "popen",
                "remove",
                "unlink",
                "rmdir",
                "mkdir",
                "write",
                "connect",
                "request",
            }:
                return False, f"unsafe_test_attribute: {node.attr}"

    return True, ""


def run_code_tests(
    code: str,
    tests: list[dict],
    timeout_seconds: float = 3.0,
) -> tuple[bool, int, str]:
    safe, reason = validate_code_safety(code)
    if not safe:
        return False, 0, reason
    safe_tests, test_reason = validate_tests_safety(tests)
    if not safe_tests:
        return False, 0, test_reason

    runner = textwrap.dedent(
        """
        import json
        import math
        import sys

        code = json.loads(sys.argv[1])
        tests = json.loads(sys.argv[2])

        def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name != "math":
                raise ImportError(f"blocked import: {name}")
            return math

        safe_builtins = {
            "__import__": safe_import,
            "abs": abs,
            "all": all,
            "any": any,
            "bool": bool,
            "dict": dict,
            "enumerate": enumerate,
            "filter": filter,
            "float": float,
            "int": int,
            "len": len,
            "list": list,
            "map": map,
            "max": max,
            "min": min,
            "range": range,
            "reversed": reversed,
            "round": round,
            "set": set,
            "sorted": sorted,
            "str": str,
            "sum": sum,
            "tuple": tuple,
            "zip": zip,
            "True": True,
            "False": False,
            "None": None,
        }
        namespace = {"__builtins__": safe_builtins, "math": math}

        def same_value(left, right):
            if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                return abs(float(left) - float(right)) <= 1e-9
            return left == right

        try:
            exec(code, namespace)
            passed = 0
            failure = ""
            for test in tests:
                try:
                    assertion = test.get("assert") or test.get("raw_assert")
                    if assertion:
                        exec(assertion, {"__builtins__": safe_builtins}, namespace)
                        passed += 1
                        continue

                    result = eval(test["call"], {"__builtins__": safe_builtins}, namespace)
                    if same_value(result, test["expected"]):
                        passed += 1
                        continue

                    failure = (
                        f"{test['call']} -> {result!r}, "
                        f"expected {test['expected']!r}"
                    )
                    break
                except AssertionError:
                    failure = assertion or "assertion_failed"
                    break
                except Exception as error:
                    label = test.get("call") or test.get("assert") or test.get("raw_assert")
                    failure = f"{label} raised {type(error).__name__}: {error}"
                    break
            print(json.dumps({"passed_tests": passed, "failure": failure}))
        except Exception as error:
            print(json.dumps({"passed_tests": 0, "failure": f"{type(error).__name__}: {error}"}))
            sys.exit(1)
        """
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        runner_path = Path(temp_dir) / "runner.py"
        runner_path.write_text(runner, encoding="utf-8")
        completed = subprocess.run(
            [
                sys.executable,
                "-I",
                str(runner_path),
                json.dumps(code),
                json.dumps(tests),
            ],
            capture_output=True,
            env={"PYTHONIOENCODING": "utf-8"},
            text=True,
            timeout=timeout_seconds,
        )

    stdout = completed.stdout.strip()
    result = {}
    if stdout:
        try:
            result = json.loads(stdout.splitlines()[-1])
        except json.JSONDecodeError:
            result = {}

    passed_tests = int(result.get("passed_tests", 0))
    failure = result.get("failure", "")
    if completed.returncode != 0 and not failure:
        failure = completed.stderr.strip() or stdout

    passed = completed.returncode == 0 and passed_tests == len(tests)
    return passed, passed_tests, failure[:500]


def evaluate_code(task: dict, generated_text: str) -> dict:
    function_name = task["function_name"]
    test_count = len(task.get("tests", []))
    code = extract_python_code(generated_text, function_name)
    if not code:
        return {
            "inferred_answer": "",
            "inferred_label": "",
            "passed": False,
            "score": 0.0,
            "error": "code_extraction_failed",
            "failure_reason": "code_extraction_failed",
            "test_count": test_count,
            "passed_tests": 0,
        }

    try:
        passed, passed_tests, error = run_code_tests(code, task["tests"])
    except subprocess.TimeoutExpired:
        passed, passed_tests, error = False, 0, "code_execution_timeout"
    except Exception as error:
        passed, passed_tests, error = False, 0, f"code_execution_error: {error}"

    return {
        "inferred_answer": "",
        "inferred_label": "",
        "passed": passed,
        "score": passed_tests / test_count if test_count else 0.0,
        "error": error,
        "failure_reason": "" if passed else error or "code_tests_failed",
        "test_count": test_count,
        "passed_tests": passed_tests,
    }


def evaluate_task(task: dict, generated_text: str) -> dict:
    category = task["category"]

    if category == "math":
        return evaluate_math(task, generated_text)
    if category == "logic":
        return evaluate_logic(task, generated_text)
    if category == "code":
        return evaluate_code(task, generated_text)

    return {
        "inferred_answer": "",
        "inferred_label": "",
        "passed": False,
        "score": 0.0,
        "error": f"unsupported_category: {category}",
        "failure_reason": f"unsupported_category: {category}",
        "test_count": 0,
        "passed_tests": 0,
    }


def _evaluate_with_profile(
    task: dict,
    generated_text: str,
    runtime_profiler: RuntimeProfiler | None,
) -> dict:
    if runtime_profiler is None:
        return evaluate_task(task, generated_text)

    with runtime_profiler.timed("evaluation_time_seconds"):
        return evaluate_task(task, generated_text)


def _profile_metrics(
    runtime_profiler: RuntimeProfiler | None,
    generated_tokens: int,
) -> dict:
    if runtime_profiler is None:
        return {}

    return runtime_profiler.summary(generated_tokens=generated_tokens)


def _task_fields(task: dict) -> dict:
    return {
        "source": task.get("source", ""),
        "difficulty": task.get("difficulty", ""),
        "answer_type": task.get("answer_type", ""),
        "expected_answer": task.get("expected_answer", ""),
        "expected_label": task.get("expected_label", ""),
        "function_name": task.get("function_name", ""),
    }


def _build_row(
    task: dict,
    mode: str,
    selected_mode: str,
    generated_text: str,
    evaluation: dict,
    estimated_saved: float,
    expensive_model_calls: int,
    model_metadata: dict,
    max_new_tokens: int,
    generated_tokens: int = 0,
    timing_stats: dict | None = None,
    profile_metrics: dict | None = None,
    prompt_router_info: dict | None = None,
) -> dict:
    timing_stats = timing_stats or {}
    profile_metrics = profile_metrics or {}
    prompt_router_info = prompt_router_info or {}
    matched_features = prompt_router_info.get("matched_features", [])
    if isinstance(matched_features, (list, tuple)):
        matched_features = "|".join(str(feature) for feature in matched_features)
    total_time_seconds = timing_stats.get("total_time_seconds_avg", "")
    tokens_per_second = timing_stats.get("tokens_per_second_avg", "")

    return {
        "task_id": task["id"],
        "category": task["category"],
        "difficulty": task.get("difficulty", ""),
        "mode": mode,
        "selected_mode": selected_mode,
        "prompt_router_reason": prompt_router_info.get("reason", ""),
        "prompt_router_matched_features": matched_features,
        "cheap_model_name": model_metadata["cheap_model_name"],
        "expensive_model_name": model_metadata["expensive_model_name"],
        "device": model_metadata["device"],
        "cheap_device": model_metadata["cheap_device"],
        "expensive_device": model_metadata["expensive_device"],
        "torch_dtype": model_metadata["torch_dtype"],
        "prompt_format": model_metadata["prompt_format"],
        "effective_prompt_format_cheap": model_metadata[
            "effective_prompt_format_cheap"
        ],
        "effective_prompt_format_expensive": model_metadata[
            "effective_prompt_format_expensive"
        ],
        "max_new_tokens": max_new_tokens,
        "generated_text": generated_text,
        **_task_fields(task),
        "inferred_answer": evaluation["inferred_answer"],
        "inferred_label": evaluation["inferred_label"],
        "test_count": evaluation["test_count"],
        "passed_tests": evaluation["passed_tests"],
        "passed": evaluation["passed"],
        "score": evaluation["score"],
        "failure_reason": evaluation["failure_reason"],
        "error": evaluation["error"],
        "estimated_saved_percent": estimated_saved,
        "expensive_model_calls": expensive_model_calls,
        "total_time_seconds": total_time_seconds,
        "total_time_seconds_avg": timing_stats.get("total_time_seconds_avg", ""),
        "total_time_seconds_std": timing_stats.get("total_time_seconds_std", ""),
        "total_time_seconds_min": timing_stats.get("total_time_seconds_min", ""),
        "total_time_seconds_max": timing_stats.get("total_time_seconds_max", ""),
        "generated_tokens": generated_tokens,
        "tokens_per_second": tokens_per_second,
        "tokens_per_second_avg": timing_stats.get("tokens_per_second_avg", ""),
        "tokens_per_second_std": timing_stats.get("tokens_per_second_std", ""),
        **profile_metrics,
    }


def _generation_from_summary(summary: dict) -> tuple[str, float, int]:
    return (
        summary["generated_text"],
        summary["estimated_saved_percent"],
        summary["expensive_model_calls"],
    )


def _normalize_devices(devices) -> list[str]:
    if isinstance(devices, (list, tuple, set)):
        values = [str(device) for device in devices]
    else:
        values = [str(devices)]
    unique = []
    for device in values:
        if device not in unique:
            unique.append(device)
    return unique


def _sync_if_cuda(devices):
    for device in _normalize_devices(devices):
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize(device)


def _measure_generation(devices, generation_fn):
    _sync_if_cuda(devices)
    start = time.perf_counter()
    result = generation_fn()
    _sync_if_cuda(devices)
    elapsed = time.perf_counter() - start
    return result, elapsed


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0

    mean = _mean(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return variance ** 0.5


def _median(values: list[float]) -> float:
    return _percentile(values, 0.50)


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0

    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _empty_timing_stats() -> dict:
    return {
        "total_time_seconds_avg": "",
        "total_time_seconds_std": "",
        "total_time_seconds_min": "",
        "total_time_seconds_max": "",
        "tokens_per_second_avg": "",
        "tokens_per_second_std": "",
    }


def _build_timing_stats(times: list[float], token_counts: list[int]) -> dict:
    if not times:
        return _empty_timing_stats()

    tokens_per_second_values = [
        token_count / elapsed if elapsed > 0 else 0.0
        for token_count, elapsed in zip(token_counts, times)
    ]

    return {
        "total_time_seconds_avg": _mean(times),
        "total_time_seconds_std": _std(times),
        "total_time_seconds_min": min(times),
        "total_time_seconds_max": max(times),
        "tokens_per_second_avg": _mean(tokens_per_second_values),
        "tokens_per_second_std": _std(tokens_per_second_values),
    }


def _run_profiled_generation(generation_fn, runtime_profiler):
    if runtime_profiler is None:
        return generation_fn(None)

    with runtime_profiler.timed("total_generation_time_seconds"):
        return generation_fn(runtime_profiler)


def _run_generation_series(
    device: str,
    generation_fn,
    token_count_fn,
    include_latency: bool,
    warmup_runs: int,
    measured_runs: int,
    progress_prefix: str = "",
    measurement_devices=None,
    runtime_profiler: RuntimeProfiler | None = None,
):
    if not include_latency:
        return _run_profiled_generation(generation_fn, runtime_profiler), _empty_timing_stats()

    warmup_total = max(0, warmup_runs)
    for run_index in range(1, warmup_total + 1):
        if progress_prefix:
            print(
                f"{progress_prefix} | warmup {run_index}/{warmup_total}",
                flush=True,
            )
        generation_fn(None)

    result = None
    times = []
    token_counts = []
    measured_total = max(1, measured_runs)
    for run_index in range(1, measured_total + 1):
        if progress_prefix:
            print(
                f"{progress_prefix} | measured {run_index}/{measured_total}",
                flush=True,
            )
        measured_runtime_profiler = runtime_profiler if result is None else None
        measured_result, elapsed = _measure_generation(
            measurement_devices or device,
            lambda: _run_profiled_generation(
                generation_fn,
                measured_runtime_profiler,
            ),
        )
        if result is None:
            result = measured_result
        times.append(elapsed)
        token_counts.append(int(token_count_fn(measured_result)))

    return result, _build_timing_stats(times, token_counts)


def _run_prompt_router_generation(
    prompt: str,
    router_mode: str,
    router_info: dict,
    cheap_model,
    expensive_model,
    cheap_tokenizer,
    expensive_tokenizer,
    cheap_device: str,
    expensive_device: str,
    max_new_tokens: int,
    temperature: float,
    prompt_format: str,
    include_latency: bool,
    warmup_runs: int,
    measured_runs: int,
    progress_prefix: str,
    runtime_profiler: RuntimeProfiler | None,
    measurement_device: str,
):
    selected_mode = router_info["selected_mode"]

    if selected_mode == "cheap_only":
        result, timing = _run_generation_series(
            measurement_device,
            lambda runtime_profiler=None: generate_greedy_with_model(
                prompt=prompt,
                model=cheap_model,
                tokenizer=cheap_tokenizer,
                device=cheap_device,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                prompt_format=prompt_format,
                runtime_profiler=runtime_profiler,
                model_role="cheap",
            ),
            lambda generation_result: generation_result[1],
            include_latency=include_latency,
            warmup_runs=warmup_runs,
            measured_runs=measured_runs,
            progress_prefix=progress_prefix,
            measurement_devices=cheap_device,
            runtime_profiler=runtime_profiler,
        )
        generated_text, generated_tokens = result
        saved = estimated_saved_percent(
            total_generated_tokens=generated_tokens,
            cheap_calls=generated_tokens,
            expensive_calls=0,
        )
        expensive_calls = 0
    elif selected_mode == "expensive_only":
        result, timing = _run_generation_series(
            measurement_device,
            lambda runtime_profiler=None: generate_greedy_with_model(
                prompt=prompt,
                model=expensive_model,
                tokenizer=expensive_tokenizer,
                device=expensive_device,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                prompt_format=prompt_format,
                runtime_profiler=runtime_profiler,
                model_role="expensive",
            ),
            lambda generation_result: generation_result[1],
            include_latency=include_latency,
            warmup_runs=warmup_runs,
            measured_runs=measured_runs,
            progress_prefix=progress_prefix,
            measurement_devices=expensive_device,
            runtime_profiler=runtime_profiler,
        )
        generated_text, generated_tokens = result
        saved = 0.0
        expensive_calls = generated_tokens
    else:
        raise ValueError(
            f"{router_mode} selected unsupported mode: {selected_mode}"
        )

    return generated_text, generated_tokens, timing, saved, expensive_calls


def run_task_evaluation(
    dataset_path: str | Path = "data/eval_tasks.jsonl",
    cheap_model_name: str = AdaptiveGenerationConfig.cheap_model_name,
    expensive_model_name: str = AdaptiveGenerationConfig.expensive_model_name,
    max_new_tokens: int = 80,
    temperature: float = 0.7,
    device: str = "auto",
    torch_dtype: str = "auto",
    prompt_format: str = "auto",
    cheap_device: str | None = None,
    expensive_device: str | None = None,
    models=None,
    include_latency: bool = False,
    warmup_runs: int = 0,
    measured_runs: int = 1,
    limit: int | None = None,
    categories: list[str] | str | None = None,
    difficulties: list[str] | str | None = None,
    modes: list[str] | str | None = None,
    profile_runtime: bool = False,
    prompt_router_model: str | Path | None = None,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    tasks = filter_eval_tasks(
        load_eval_tasks(dataset_path),
        categories=categories,
        difficulties=difficulties,
        limit=limit,
    )
    selected_modes = resolve_task_modes(modes)
    selected_mode_set = set(selected_modes)
    prompt_router_ml_model = None
    if "prompt_router_ml_v1" in selected_mode_set:
        if prompt_router_model is None:
            raise ValueError(
                "prompt_router_ml_v1 requires --prompt-router-model "
                "pointing to a trained .joblib file."
            )
        prompt_router_ml_model = load_prompt_router_ml_model(prompt_router_model)

    if models is None:
        config = AdaptiveGenerationConfig(
            cheap_model_name=cheap_model_name,
            expensive_model_name=expensive_model_name,
            device=device,
            cheap_device=cheap_device,
            expensive_device=expensive_device,
            torch_dtype=torch_dtype,
            prompt_format=prompt_format,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        cheap_model, expensive_model, tokenizer, device = load_adaptive_models(
            config
        )
    else:
        cheap_model, expensive_model, tokenizer, device = models

    cheap_runtime = get_model_runtime_metadata(cheap_model, fallback_device=device)
    expensive_runtime = get_model_runtime_metadata(
        expensive_model,
        fallback_device=device,
    )
    if cheap_runtime["torch_dtype"] != expensive_runtime["torch_dtype"]:
        raise ValueError(
            "cheap_model and expensive_model must use the same dtype. "
            f"cheap={cheap_runtime}, expensive={expensive_runtime}"
        )

    model_metadata = {
        "cheap_model_name": cheap_model_name,
        "expensive_model_name": expensive_model_name,
        "device": (
            cheap_runtime["device"]
            if cheap_runtime["device"] == expensive_runtime["device"]
            else "split"
        ),
        "cheap_device": cheap_runtime["device"],
        "expensive_device": expensive_runtime["device"],
        "torch_dtype": cheap_runtime["torch_dtype"],
        **prompt_format_metadata(tokenizer, prompt_format),
    }
    cheap_tokenizer = get_cheap_tokenizer(tokenizer)
    expensive_tokenizer = get_expensive_tokenizer(tokenizer)
    rows = []

    total_tasks = len(tasks)

    for task_index, task in enumerate(tasks, start=1):
        prompt = task["prompt"]
        prompt_type = classify_prompt(prompt)
        hybrid_selected_mode = choose_mode(prompt_type, prompt)
        prompt_router_infos = {
            "prompt_router_v1": classify_prompt_router_v1(prompt),
            "prompt_router_v2": classify_prompt_router_v2(prompt),
        }
        if prompt_router_ml_model is not None:
            prompt_router_infos["prompt_router_ml_v1"] = classify_prompt_router_ml_v1(
                prompt,
                prompt_router_ml_model,
            )
        mode_summaries = {}

        def progress_prefix(mode: str) -> str:
            return f"[task {task_index}/{total_tasks}] {task['id']} | mode {mode}"

        if "expensive_only" in selected_mode_set:
            expensive_profiler = maybe_profiler(profile_runtime)
            (expensive_text, expensive_tokens), expensive_timing = (
                _run_generation_series(
                    device,
                    lambda runtime_profiler=None: generate_greedy_with_model(
                        prompt=prompt,
                        model=expensive_model,
                        tokenizer=expensive_tokenizer,
                        device=expensive_runtime["device"],
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        prompt_format=prompt_format,
                        runtime_profiler=runtime_profiler,
                        model_role="expensive",
                    ),
                    lambda result: result[1],
                    include_latency=include_latency,
                    warmup_runs=warmup_runs,
                    measured_runs=measured_runs,
                    progress_prefix=progress_prefix("expensive_only"),
                    measurement_devices=expensive_runtime["device"],
                    runtime_profiler=expensive_profiler,
                )
            )
            evaluation = _evaluate_with_profile(task, expensive_text, expensive_profiler)
            rows.append(
                _build_row(
                    task=task,
                    mode="expensive_only",
                    selected_mode="",
                    generated_text=expensive_text,
                    evaluation=evaluation,
                    estimated_saved=0.0,
                    expensive_model_calls=expensive_tokens,
                    model_metadata=model_metadata,
                    max_new_tokens=max_new_tokens,
                    generated_tokens=expensive_tokens,
                    timing_stats=expensive_timing,
                    profile_metrics=_profile_metrics(
                        expensive_profiler,
                        expensive_tokens,
                    ),
                )
            )

        if "cheap_only" in selected_mode_set:
            cheap_profiler = maybe_profiler(profile_runtime)
            (cheap_text, cheap_tokens), cheap_timing = _run_generation_series(
                device,
                lambda runtime_profiler=None: generate_greedy_with_model(
                    prompt=prompt,
                    model=cheap_model,
                    tokenizer=cheap_tokenizer,
                    device=cheap_runtime["device"],
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    prompt_format=prompt_format,
                    runtime_profiler=runtime_profiler,
                    model_role="cheap",
                ),
                lambda result: result[1],
                include_latency=include_latency,
                warmup_runs=warmup_runs,
                measured_runs=measured_runs,
                progress_prefix=progress_prefix("cheap_only"),
                measurement_devices=cheap_runtime["device"],
                runtime_profiler=cheap_profiler,
            )
            cheap_saved = estimated_saved_percent(
                total_generated_tokens=cheap_tokens,
                cheap_calls=cheap_tokens,
                expensive_calls=0,
            )
            evaluation = _evaluate_with_profile(task, cheap_text, cheap_profiler)
            rows.append(
                _build_row(
                    task=task,
                    mode="cheap_only",
                    selected_mode="",
                    generated_text=cheap_text,
                    evaluation=evaluation,
                    estimated_saved=cheap_saved,
                    expensive_model_calls=0,
                    model_metadata=model_metadata,
                    max_new_tokens=max_new_tokens,
                    generated_tokens=cheap_tokens,
                    timing_stats=cheap_timing,
                    profile_metrics=_profile_metrics(cheap_profiler, cheap_tokens),
                )
            )

        for router_mode, prompt_router_info in prompt_router_infos.items():
            if router_mode not in selected_mode_set:
                continue

            router_profiler = maybe_profiler(profile_runtime)
            (
                router_text,
                router_tokens,
                router_timing,
                router_saved,
                router_expensive_calls,
            ) = _run_prompt_router_generation(
                prompt=prompt,
                router_mode=router_mode,
                router_info=prompt_router_info,
                cheap_model=cheap_model,
                expensive_model=expensive_model,
                cheap_tokenizer=cheap_tokenizer,
                expensive_tokenizer=expensive_tokenizer,
                cheap_device=cheap_runtime["device"],
                expensive_device=expensive_runtime["device"],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                prompt_format=prompt_format,
                include_latency=include_latency,
                warmup_runs=warmup_runs,
                measured_runs=measured_runs,
                progress_prefix=progress_prefix(router_mode),
                runtime_profiler=router_profiler,
                measurement_device=device,
            )
            evaluation = _evaluate_with_profile(
                task,
                router_text,
                router_profiler,
            )
            rows.append(
                _build_row(
                    task=task,
                    mode=router_mode,
                    selected_mode=prompt_router_info["selected_mode"],
                    generated_text=router_text,
                    evaluation=evaluation,
                    estimated_saved=router_saved,
                    expensive_model_calls=router_expensive_calls,
                    model_metadata=model_metadata,
                    max_new_tokens=max_new_tokens,
                    generated_tokens=router_tokens,
                    timing_stats=router_timing,
                    profile_metrics=_profile_metrics(
                        router_profiler,
                        router_tokens,
                    ),
                    prompt_router_info=prompt_router_info,
                )
            )

        for mode in (
            "adaptive_calibrated",
            "adaptive_guarded_v3",
            "adaptive_code_quality",
            "speculative_adaptive",
        ):
            needed_for_hybrid = (
                not include_latency
                and not profile_runtime
                and "hybrid" in selected_mode_set
                and mode == hybrid_selected_mode
            )
            if mode not in selected_mode_set and not needed_for_hybrid:
                continue

            mode_profiler = maybe_profiler(profile_runtime)
            summary, mode_timing = _run_generation_series(
                device,
                lambda runtime_profiler=None, mode=mode: generate_with_mode(
                    prompt=prompt,
                    mode=mode,
                    cheap_model=cheap_model,
                    expensive_model=expensive_model,
                    tokenizer=tokenizer,
                    device=cheap_runtime["device"],
                    cheap_model_name=cheap_model_name,
                    expensive_model_name=expensive_model_name,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    prompt_format=prompt_format,
                    runtime_profiler=runtime_profiler,
                ),
                lambda result: result.get("total_generated_tokens", 0),
                include_latency=include_latency,
                warmup_runs=warmup_runs,
                measured_runs=measured_runs,
                progress_prefix=progress_prefix(mode),
                measurement_devices=[
                    cheap_runtime["device"],
                    expensive_runtime["device"],
                ],
                runtime_profiler=mode_profiler,
            )
            mode_summaries[mode] = summary
            if mode in selected_mode_set:
                generated_text, saved, expensive_calls = _generation_from_summary(
                    summary
                )
                generated_token_count = summary.get("total_generated_tokens", 0)
                evaluation = _evaluate_with_profile(
                    task,
                    generated_text,
                    mode_profiler,
                )
                rows.append(
                    _build_row(
                        task=task,
                        mode=mode,
                        selected_mode="",
                        generated_text=generated_text,
                        evaluation=evaluation,
                        estimated_saved=saved,
                        expensive_model_calls=expensive_calls,
                        model_metadata=model_metadata,
                        max_new_tokens=max_new_tokens,
                        generated_tokens=generated_token_count,
                        timing_stats=mode_timing,
                        profile_metrics=_profile_metrics(
                            mode_profiler,
                            generated_token_count,
                        ),
                    )
                )

        if "hybrid" not in selected_mode_set:
            continue

        if include_latency or profile_runtime:
            hybrid_profiler = maybe_profiler(profile_runtime)
            hybrid_summary, hybrid_timing = _run_generation_series(
                device,
                lambda runtime_profiler=None: generate_with_mode(
                    prompt=prompt,
                    mode=hybrid_selected_mode,
                    cheap_model=cheap_model,
                    expensive_model=expensive_model,
                    tokenizer=tokenizer,
                    device=cheap_runtime["device"],
                    cheap_model_name=cheap_model_name,
                    expensive_model_name=expensive_model_name,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    prompt_format=prompt_format,
                    runtime_profiler=runtime_profiler,
                ),
                lambda result: result.get("total_generated_tokens", 0),
                include_latency=include_latency,
                warmup_runs=warmup_runs,
                measured_runs=measured_runs,
                progress_prefix=progress_prefix("hybrid"),
                measurement_devices=[
                    cheap_runtime["device"],
                    expensive_runtime["device"],
                ],
                runtime_profiler=hybrid_profiler,
            )
        else:
            hybrid_profiler = None
            if hybrid_selected_mode in mode_summaries:
                hybrid_summary = mode_summaries[hybrid_selected_mode]
            else:
                hybrid_summary = generate_with_mode(
                    prompt=prompt,
                    mode=hybrid_selected_mode,
                    cheap_model=cheap_model,
                    expensive_model=expensive_model,
                    tokenizer=tokenizer,
                    device=cheap_runtime["device"],
                    cheap_model_name=cheap_model_name,
                    expensive_model_name=expensive_model_name,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    prompt_format=prompt_format,
                )
            hybrid_timing = _empty_timing_stats()
        generated_text, saved, expensive_calls = _generation_from_summary(
            hybrid_summary
        )
        generated_token_count = hybrid_summary.get("total_generated_tokens", 0)
        evaluation = _evaluate_with_profile(task, generated_text, hybrid_profiler)
        rows.append(
            _build_row(
                task=task,
                mode="hybrid",
                selected_mode=hybrid_selected_mode,
                generated_text=generated_text,
                evaluation=evaluation,
                estimated_saved=saved,
                expensive_model_calls=expensive_calls,
                model_metadata=model_metadata,
                max_new_tokens=max_new_tokens,
                generated_tokens=generated_token_count,
                timing_stats=hybrid_timing,
                profile_metrics=_profile_metrics(
                    hybrid_profiler,
                    generated_token_count,
                ),
            )
        )

    return (
        rows,
        summarize_task_evaluation(rows),
        summarize_task_evaluation_by_difficulty(rows),
        summarize_task_evaluation_overall(rows),
    )


def _wilson_interval(passed: int, count: int, z: float = 1.96) -> tuple[float, float]:
    if count <= 0:
        return 0.0, 0.0

    phat = passed / count
    denominator = 1 + z**2 / count
    center = (phat + z**2 / (2 * count)) / denominator
    margin = (
        z
        * ((phat * (1 - phat) + z**2 / (4 * count)) / count) ** 0.5
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def _summary_from_group(group: list[dict], extra: dict) -> dict:
    count = len(group)
    first_row = group[0] if group else {}
    passed = sum(1 for row in group if row["passed"])
    pass_rate = passed / count if count else 0.0
    ci_low, ci_high = _wilson_interval(passed, count)
    score_sum = sum(float(row["score"]) for row in group)
    saved_sum = sum(float(row["estimated_saved_percent"]) for row in group)
    calls_sum = sum(float(row["expensive_model_calls"]) for row in group)

    return {
        **extra,
        "count": count,
        "pass_rate": pass_rate,
        "pass_rate_ci95_low": ci_low,
        "pass_rate_ci95_high": ci_high,
        "avg_score": score_sum / count if count else 0.0,
        "avg_estimated_saved_percent": saved_sum / count if count else 0.0,
        "avg_expensive_model_calls": calls_sum / count if count else 0.0,
        "cheap_model_name": first_row.get("cheap_model_name", ""),
        "expensive_model_name": first_row.get("expensive_model_name", ""),
        "device": first_row.get("device", ""),
        "cheap_device": first_row.get("cheap_device", first_row.get("device", "")),
        "expensive_device": first_row.get(
            "expensive_device",
            first_row.get("device", ""),
        ),
        "torch_dtype": first_row.get("torch_dtype", ""),
        "prompt_format": first_row.get("prompt_format", ""),
    }


def summarize_task_evaluation(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["category"], row["mode"])].append(row)

    summary_rows = []
    mode_order = {mode: index for index, mode in enumerate(TASK_MODES)}
    for (category, mode), group in sorted(
        grouped.items(),
        key=lambda item: (item[0][0], mode_order.get(item[0][1], 999)),
    ):
        summary_rows.append(
            _summary_from_group(
                group,
                {
                    "category": category,
                    "mode": mode,
                },
            )
        )

    return summary_rows


def summarize_task_evaluation_by_difficulty(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["category"], row["difficulty"], row["mode"])].append(row)

    difficulty_order = {"easy": 0, "medium": 1, "hard": 2}
    mode_order = {mode: index for index, mode in enumerate(TASK_MODES)}
    summary_rows = []
    for (category, difficulty, mode), group in sorted(
        grouped.items(),
        key=lambda item: (
            item[0][0],
            difficulty_order.get(item[0][1], 999),
            mode_order.get(item[0][2], 999),
        ),
    ):
        summary_rows.append(
            _summary_from_group(
                group,
                {
                    "category": category,
                    "difficulty": difficulty,
                    "mode": mode,
                },
            )
        )

    return summary_rows


def summarize_task_evaluation_overall(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["mode"]].append(row)

    mode_order = {mode: index for index, mode in enumerate(TASK_MODES)}
    summary_rows = []
    for mode, group in sorted(
        grouped.items(),
        key=lambda item: mode_order.get(item[0], 999),
    ):
        summary_rows.append(
            _summary_from_group(
                group,
                {
                    "mode": mode,
                },
            )
        )

    return summary_rows


def _as_float(value, default: float = 0.0) -> float:
    if value == "" or value is None:
        return default
    return float(value)


def build_task_quality_latency_report(rows: list[dict]) -> list[dict]:
    baselines = {}
    for row in rows:
        if row["mode"] == "expensive_only":
            baselines[row["task_id"]] = _as_float(
                row.get("total_time_seconds_avg", row["total_time_seconds"])
            )

    report_rows = []
    for row in rows:
        total_time = _as_float(row["total_time_seconds"])
        total_time_avg = _as_float(row.get("total_time_seconds_avg", total_time))
        total_time_std = _as_float(row.get("total_time_seconds_std", 0.0))
        total_time_min = _as_float(row.get("total_time_seconds_min", total_time_avg))
        total_time_max = _as_float(row.get("total_time_seconds_max", total_time_avg))
        tokens_per_second_avg = _as_float(
            row.get("tokens_per_second_avg", row["tokens_per_second"])
        )
        tokens_per_second_std = _as_float(row.get("tokens_per_second_std", 0.0))
        baseline = baselines.get(row["task_id"], 0.0)
        if baseline > 0 and total_time_avg > 0:
            speedup = 100 * (1 - total_time_avg / baseline)
            speedup_std = 100 * total_time_std / baseline
        else:
            speedup = 0.0
            speedup_std = 0.0

        report_rows.append(
            {
                "task_id": row["task_id"],
                "category": row["category"],
                "difficulty": row["difficulty"],
                "mode": row["mode"],
                "selected_mode": row["selected_mode"],
                "passed": row["passed"],
                "score": row["score"],
                "total_time_seconds": total_time_avg,
                "total_time_seconds_avg": total_time_avg,
                "total_time_seconds_std": total_time_std,
                "total_time_seconds_min": total_time_min,
                "total_time_seconds_max": total_time_max,
                "expensive_only_seconds": baseline,
                "expensive_only_seconds_avg": baseline,
                "real_speedup_vs_expensive_percent": speedup,
                "real_speedup_vs_expensive_percent_avg": speedup,
                "real_speedup_vs_expensive_percent_std": speedup_std,
                "generated_tokens": row["generated_tokens"],
                "tokens_per_second": tokens_per_second_avg,
                "tokens_per_second_avg": tokens_per_second_avg,
                "tokens_per_second_std": tokens_per_second_std,
                "estimated_saved_percent": row["estimated_saved_percent"],
                "expensive_model_calls": row["expensive_model_calls"],
                "cheap_model_name": row["cheap_model_name"],
                "expensive_model_name": row["expensive_model_name"],
                "device": row["device"],
                "cheap_device": row.get("cheap_device", row["device"]),
                "expensive_device": row.get("expensive_device", row["device"]),
                "torch_dtype": row["torch_dtype"],
                "prompt_format": row["prompt_format"],
                "failure_reason": row["failure_reason"],
                "generated_text": row["generated_text"],
            }
        )

    return report_rows


def summarize_task_quality_latency(
    report_rows: list[dict],
    group_fields: tuple[str, ...] = ("mode",),
) -> list[dict]:
    grouped = defaultdict(list)
    for row in report_rows:
        grouped[tuple(row[field] for field in group_fields)].append(row)

    mode_order = {mode: index for index, mode in enumerate(TASK_MODES)}
    difficulty_order = {"easy": 0, "medium": 1, "hard": 2}

    def sort_key(item):
        key = item[0]
        parts = []
        for field, value in zip(group_fields, key):
            if field == "mode":
                parts.append(mode_order.get(value, 999))
            elif field == "difficulty":
                parts.append(difficulty_order.get(value, 999))
            else:
                parts.append(value)
        return tuple(parts)

    summary_rows = []
    for key, group in sorted(grouped.items(), key=sort_key):
        count = len(group)
        first_row = group[0] if group else {}
        passed = sum(1 for row in group if row["passed"] is True or row["passed"] == "True")
        pass_rate = passed / count if count else 0.0
        ci_low, ci_high = _wilson_interval(passed, count)
        score_sum = sum(_as_float(row["score"]) for row in group)
        time_values = [_as_float(row["total_time_seconds_avg"]) for row in group]
        speedup_values = [
            _as_float(row["real_speedup_vs_expensive_percent_avg"]) for row in group
        ]
        saved_sum = sum(_as_float(row["estimated_saved_percent"]) for row in group)
        calls_sum = sum(_as_float(row["expensive_model_calls"]) for row in group)

        summary = {field: value for field, value in zip(group_fields, key)}
        summary.update(
            {
                "count": count,
                "pass_rate": pass_rate,
                "avg_pass_rate": pass_rate,
                "pass_rate_ci95_low": ci_low,
                "pass_rate_ci95_high": ci_high,
                "avg_score": score_sum / count if count else 0.0,
                "avg_total_time_seconds": _mean(time_values),
                "std_total_time_seconds": _std(time_values),
                "median_total_time_seconds": _median(time_values),
                "p25_total_time_seconds": _percentile(time_values, 0.25),
                "p75_total_time_seconds": _percentile(time_values, 0.75),
                "min_total_time_seconds": min(time_values) if time_values else 0.0,
                "max_total_time_seconds": max(time_values) if time_values else 0.0,
                "avg_real_speedup_vs_expensive_percent": (
                    _mean(speedup_values)
                ),
                "median_real_speedup_vs_expensive_percent": _median(speedup_values),
                "avg_estimated_saved_percent": saved_sum / count if count else 0.0,
                "avg_expensive_model_calls": calls_sum / count if count else 0.0,
                "cheap_model_name": first_row.get("cheap_model_name", ""),
                "expensive_model_name": first_row.get("expensive_model_name", ""),
                "device": first_row.get("device", ""),
                "cheap_device": first_row.get(
                    "cheap_device",
                    first_row.get("device", ""),
                ),
                "expensive_device": first_row.get(
                    "expensive_device",
                    first_row.get("device", ""),
                ),
                "torch_dtype": first_row.get("torch_dtype", ""),
                "prompt_format": first_row.get("prompt_format", ""),
            }
        )
        summary_rows.append(summary)

    return summary_rows


def build_task_quality_latency_outputs(
    rows: list[dict],
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    report_rows = build_task_quality_latency_report(rows)
    summary_rows = summarize_task_quality_latency(report_rows, ("mode",))
    by_category_rows = summarize_task_quality_latency(
        report_rows,
        ("category", "mode"),
    )
    by_difficulty_rows = summarize_task_quality_latency(
        report_rows,
        ("difficulty", "mode"),
    )

    return report_rows, summary_rows, by_category_rows, by_difficulty_rows


def build_runtime_profile_rows(rows: list[dict]) -> list[dict]:
    profile_rows = []
    for row in rows:
        if "total_generation_time_seconds" not in row:
            continue

        profile_rows.append(
            {
                "task_id": row["task_id"],
                "category": row["category"],
                "difficulty": row["difficulty"],
                "mode": row["mode"],
                "selected_mode": row["selected_mode"],
                "prompt_router_reason": row.get("prompt_router_reason", ""),
                "prompt_router_matched_features": row.get(
                    "prompt_router_matched_features",
                    "",
                ),
                "cheap_model_name": row["cheap_model_name"],
                "expensive_model_name": row["expensive_model_name"],
                "device": row["device"],
                "cheap_device": row["cheap_device"],
                "expensive_device": row["expensive_device"],
                "torch_dtype": row["torch_dtype"],
                "prompt_format": row["prompt_format"],
                "passed": row["passed"],
                "score": row["score"],
                "estimated_saved_percent": row["estimated_saved_percent"],
                "expensive_model_calls": row["expensive_model_calls"],
                "generated_tokens": row["generated_tokens"],
                **{field: row.get(field, 0.0) for field in PROFILE_FIELDS},
            }
        )

    return profile_rows


def summarize_runtime_profile_rows(profile_rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in profile_rows:
        grouped[row["mode"]].append(row)

    mode_order = {mode: index for index, mode in enumerate(TASK_MODES)}
    summary_rows = []
    for mode, group in sorted(
        grouped.items(),
        key=lambda item: mode_order.get(item[0], 999),
    ):
        count = len(group)
        first_row = group[0]
        passed = sum(1 for row in group if row["passed"] is True or row["passed"] == "True")
        summary = {
            "mode": mode,
            "count": count,
            "pass_rate": passed / count if count else 0.0,
            "cheap_model_name": first_row.get("cheap_model_name", ""),
            "expensive_model_name": first_row.get("expensive_model_name", ""),
            "device": first_row.get("device", ""),
            "cheap_device": first_row.get("cheap_device", ""),
            "expensive_device": first_row.get("expensive_device", ""),
            "torch_dtype": first_row.get("torch_dtype", ""),
            "prompt_format": first_row.get("prompt_format", ""),
            "avg_estimated_saved_percent": _mean(
                [_as_float(row["estimated_saved_percent"]) for row in group]
            ),
            "avg_expensive_model_calls": _mean(
                [_as_float(row["expensive_model_calls"]) for row in group]
            ),
        }

        for field in TIME_FIELDS + DERIVED_PROFILE_FIELDS:
            values = [_as_float(row[field]) for row in group]
            summary[f"avg_{field}"] = _mean(values)
            summary[f"median_{field}"] = _median(values)

        for field in COUNT_FIELDS:
            values = [_as_float(row[field]) for row in group]
            summary[f"avg_{field}"] = _mean(values)
            summary[f"total_{field}"] = sum(values)

        summary_rows.append(summary)

    return summary_rows


def build_runtime_profile_outputs(
    rows: list[dict],
) -> tuple[list[dict], list[dict]]:
    profile_rows = build_runtime_profile_rows(rows)
    summary_rows = summarize_runtime_profile_rows(profile_rows)
    return profile_rows, summary_rows


def print_task_evaluation_report(summary_rows: list[dict]):
    print()
    print("Task-Specific Quality Evaluation")
    print("=" * 88)
    header = (
        f"{'category':<10} | {'mode':<22} | {'pass_rate':>9} | "
        f"{'avg_saved':>10} | {'calls':>7}"
    )
    print(header)
    print("-" * len(header))

    for row in summary_rows:
        print(
            f"{row['category']:<10} | "
            f"{row['mode']:<22} | "
            f"{row['pass_rate']:>8.2%} | "
            f"{row['avg_estimated_saved_percent']:>9.2f}% | "
            f"{row['avg_expensive_model_calls']:>7.2f}"
        )

    print("=" * 88)
    print()


def print_task_evaluation_overall_report(overall_rows: list[dict]):
    print("Overall by mode")
    print("=" * 72)
    header = (
        f"{'mode':<22} | {'pass_rate':>9} | {'avg_score':>9} | "
        f"{'avg_saved':>10} | {'calls':>7}"
    )
    print(header)
    print("-" * len(header))
    for row in overall_rows:
        print(
            f"{row['mode']:<22} | "
            f"{row['pass_rate']:>8.2%} | "
            f"{row['avg_score']:>9.3f} | "
            f"{row['avg_estimated_saved_percent']:>9.2f}% | "
            f"{row['avg_expensive_model_calls']:>7.2f}"
        )
    print("=" * 72)
    print()


def print_task_quality_latency_report(summary_rows: list[dict]):
    print("Task Quality-Latency Report")
    print("=" * 116)
    header = (
        f"{'mode':<22} | {'pass_rate':>9} | {'avg_speedup':>11} | "
        f"{'avg_saved':>10} | {'avg_calls':>9} | {'avg_time':>9} | "
        f"{'std_time':>9}"
    )
    print(header)
    print("-" * len(header))
    for row in summary_rows:
        print(
            f"{row['mode']:<22} | "
            f"{row['pass_rate']:>8.2%} | "
            f"{row['avg_real_speedup_vs_expensive_percent']:>10.2f}% | "
            f"{row['avg_estimated_saved_percent']:>9.2f}% | "
            f"{row['avg_expensive_model_calls']:>9.2f} | "
            f"{row['avg_total_time_seconds']:>9.3f} | "
            f"{row['std_total_time_seconds']:>9.3f}"
        )
    print("=" * 116)
    print()


def print_runtime_profile_report(summary_rows: list[dict]):
    print("Runtime Profile Summary")
    print("=" * 116)
    header = (
        f"{'mode':<22} | {'total':>9} | {'cheap_fwd':>9} | "
        f"{'exp_fwd':>9} | {'route_ovh':>9} | {'cheap_n':>7} | "
        f"{'exp_n':>7}"
    )
    print(header)
    print("-" * len(header))
    for row in summary_rows:
        print(
            f"{row['mode']:<22} | "
            f"{row['avg_total_generation_time_seconds']:>9.3f} | "
            f"{row['avg_cheap_forward_time_seconds']:>9.3f} | "
            f"{row['avg_expensive_forward_time_seconds']:>9.3f} | "
            f"{row['avg_routing_overhead_time_seconds']:>9.3f} | "
            f"{row['avg_number_of_cheap_forwards']:>7.2f} | "
            f"{row['avg_number_of_expensive_forwards']:>7.2f}"
        )
    print("=" * 116)
    print()


def save_task_evaluation_outputs(
    rows: list[dict],
    summary_rows: list[dict],
    difficulty_rows: list[dict],
    overall_rows: list[dict],
    output_dir: str | Path = "results",
) -> tuple[Path, Path, Path, Path]:
    output_path = Path(output_dir)
    detailed_csv = output_path / "task_evaluation.csv"
    summary_csv = output_path / "task_evaluation_summary.csv"
    difficulty_csv = output_path / "task_evaluation_by_difficulty.csv"
    overall_csv = output_path / "task_evaluation_overall.csv"

    save_csv(rows, str(detailed_csv))
    save_csv(summary_rows, str(summary_csv))
    save_csv(difficulty_rows, str(difficulty_csv))
    save_csv(overall_rows, str(overall_csv))

    return detailed_csv, summary_csv, difficulty_csv, overall_csv


def save_task_quality_latency_outputs(
    report_rows: list[dict],
    summary_rows: list[dict],
    by_category_rows: list[dict],
    by_difficulty_rows: list[dict],
    output_dir: str | Path = "results",
) -> tuple[Path, Path, Path, Path]:
    output_path = Path(output_dir)
    report_csv = output_path / "task_quality_latency_report.csv"
    summary_csv = output_path / "task_quality_latency_summary.csv"
    by_category_csv = output_path / "task_quality_latency_by_category.csv"
    by_difficulty_csv = output_path / "task_quality_latency_by_difficulty.csv"

    save_csv(report_rows, str(report_csv))
    save_csv(summary_rows, str(summary_csv))
    save_csv(by_category_rows, str(by_category_csv))
    save_csv(by_difficulty_rows, str(by_difficulty_csv))

    return report_csv, summary_csv, by_category_csv, by_difficulty_csv


def save_runtime_profile_outputs(
    profile_rows: list[dict],
    summary_rows: list[dict],
    output_dir: str | Path = "results",
) -> tuple[Path, Path]:
    output_path = Path(output_dir)
    profile_csv = output_path / "runtime_profile.csv"
    summary_csv = output_path / "runtime_profile_summary.csv"

    save_csv(profile_rows, str(profile_csv))
    save_csv(summary_rows, str(summary_csv))

    return profile_csv, summary_csv
