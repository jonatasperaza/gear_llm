import ast
import json
import re
import subprocess
import sys
import tempfile
import textwrap
import unicodedata
from collections import defaultdict
from pathlib import Path

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
from gear_llm.report import save_csv


TASK_MODES = (
    "expensive_only",
    "cheap_only",
    "adaptive_calibrated",
    "adaptive_guarded_v3",
    "speculative_adaptive",
    "hybrid",
)


def load_eval_tasks(path: str | Path) -> list[dict]:
    tasks = []
    dataset_path = Path(path)

    with dataset_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            item = json.loads(stripped)
            missing = {"id", "category", "prompt"} - set(item)
            if missing:
                fields = ", ".join(sorted(missing))
                raise ValueError(
                    f"Line {line_number} of {dataset_path} missing fields: {fields}"
                )

            category = item["category"]
            if category == "math" and "expected_answer" not in item:
                raise ValueError(f"Task {item['id']} missing expected_answer.")
            if category == "logic" and "acceptable_labels" not in item:
                raise ValueError(f"Task {item['id']} missing acceptable_labels.")
            if category == "code":
                missing_code = {"function_name", "tests"} - set(item)
                if missing_code:
                    fields = ", ".join(sorted(missing_code))
                    raise ValueError(f"Task {item['id']} missing fields: {fields}")

            tasks.append(item)

    return tasks


def _strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


def normalize_answer(text: str) -> str:
    text = _strip_accents(text.lower())
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
    text = text.replace("\\", "")
    text = re.sub(r"(?<=\d),(?=\d)", ".", text)
    text = re.sub(r"\s+", "", text)
    return text


def evaluate_math(task: dict, generated_text: str) -> dict:
    expected_answers = [task["expected_answer"]]
    expected_answers.extend(task.get("acceptable_answers", []))

    normalized_generated = normalize_answer(generated_text)
    for answer in expected_answers:
        normalized_answer = normalize_answer(str(answer))
        if normalized_answer and normalized_answer in normalized_generated:
            return {
                "inferred_answer": answer,
                "inferred_label": "",
                "passed": True,
                "score": 1.0,
                "error": "",
            }

    return {
        "inferred_answer": "",
        "inferred_label": "",
        "passed": False,
        "score": 0.0,
        "error": "",
    }


LOGIC_PATTERNS = {
    "unknown": (
        "unknown",
        "unclear",
        "not enough",
        "insufficient",
        "cannot determine",
        "can't determine",
        "indeterminate",
        "desconhecido",
        "incerto",
        "indeterminado",
        "nao e possivel determinar",
    ),
    "depends": (
        "depends",
        "it depends",
        "depending",
        "depende",
    ),
    "deny": (
        "deny",
        "denied",
        "should not",
        "not allowed",
        "not be allowed",
        "no",
        "block",
        "blocked",
        "reject",
        "rejected",
        "prohibit",
        "forbid",
        "negar",
        "negado",
        "bloquear",
        "bloqueado",
        "nao",
        "proibido",
    ),
    "allow": (
        "allow",
        "allowed",
        "yes",
        "permit",
        "permitted",
        "grant",
        "sim",
        "permitir",
        "permitido",
        "autorizar",
        "autorizado",
    ),
}


def infer_logic_label(text: str) -> str:
    normalized = _strip_accents(text.lower())
    matches = []

    for label, patterns in LOGIC_PATTERNS.items():
        for pattern in patterns:
            if re.search(rf"\b{re.escape(pattern)}\b", normalized):
                matches.append((normalized.find(pattern), label))

    if not matches:
        return "unknown"

    matches.sort(key=lambda item: item[0])
    return matches[0][1]


def evaluate_logic(task: dict, generated_text: str) -> dict:
    inferred_label = infer_logic_label(generated_text)
    acceptable = set(task.get("acceptable_labels", []))
    passed = inferred_label in acceptable

    return {
        "inferred_answer": "",
        "inferred_label": inferred_label,
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "error": "",
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


def _trim_function_block(code: str, function_name: str) -> str:
    lines = code.replace("\r\n", "\n").split("\n")
    start = None
    pattern = re.compile(rf"^\s*def\s+{re.escape(function_name)}\s*\(")

    for index, line in enumerate(lines):
        if pattern.search(line):
            start = index
            break

    if start is None:
        return ""

    block = [lines[start]]
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

    return _trim_function_block(generated_text[match.start() :], function_name)


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


def validate_code_safety(code: str) -> tuple[bool, str]:
    try:
        tree = ast.parse(code)
    except SyntaxError as error:
        return False, f"syntax_error: {error}"

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return False, "imports_are_not_allowed"
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


def run_code_tests(
    code: str,
    tests: list[dict],
    timeout_seconds: float = 3.0,
) -> tuple[bool, str]:
    safe, reason = validate_code_safety(code)
    if not safe:
        return False, reason

    runner = textwrap.dedent(
        """
        import json
        import math
        import sys

        code = json.loads(sys.argv[1])
        tests = json.loads(sys.argv[2])
        safe_builtins = {
            "abs": abs,
            "all": all,
            "any": any,
            "bool": bool,
            "dict": dict,
            "enumerate": enumerate,
            "float": float,
            "int": int,
            "len": len,
            "list": list,
            "max": max,
            "min": min,
            "range": range,
            "round": round,
            "set": set,
            "str": str,
            "sum": sum,
            "tuple": tuple,
            "True": True,
            "False": False,
            "None": None,
        }
        namespace = {"__builtins__": safe_builtins}
        exec(code, namespace)

        def same_value(left, right):
            if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                return abs(float(left) - float(right)) <= 1e-9
            return left == right

        for test in tests:
            result = eval(test["call"], {"__builtins__": safe_builtins}, namespace)
            if not same_value(result, test["expected"]):
                raise AssertionError(
                    f"{test['call']} -> {result!r}, expected {test['expected']!r}"
                )
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

    if completed.returncode != 0:
        error = completed.stderr.strip() or completed.stdout.strip()
        return False, error[:500]

    return True, ""


def evaluate_code(task: dict, generated_text: str) -> dict:
    function_name = task["function_name"]
    code = extract_python_code(generated_text, function_name)
    if not code:
        return {
            "inferred_answer": "",
            "inferred_label": "",
            "passed": False,
            "score": 0.0,
            "error": "code_extraction_failed",
        }

    try:
        passed, error = run_code_tests(code, task["tests"])
    except subprocess.TimeoutExpired:
        passed, error = False, "code_execution_timeout"
    except Exception as error:
        passed, error = False, f"code_execution_error: {error}"

    return {
        "inferred_answer": "",
        "inferred_label": "",
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "error": error,
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
    }


def _empty_task_fields(task: dict) -> dict:
    return {
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
) -> dict:
    return {
        "task_id": task["id"],
        "category": task["category"],
        "mode": mode,
        "selected_mode": selected_mode,
        "cheap_model_name": model_metadata["cheap_model_name"],
        "expensive_model_name": model_metadata["expensive_model_name"],
        "device": model_metadata["device"],
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
        **_empty_task_fields(task),
        "inferred_answer": evaluation["inferred_answer"],
        "inferred_label": evaluation["inferred_label"],
        "passed": evaluation["passed"],
        "score": evaluation["score"],
        "error": evaluation["error"],
        "estimated_saved_percent": estimated_saved,
        "expensive_model_calls": expensive_model_calls,
    }


def _generation_from_summary(summary: dict) -> tuple[str, float, int]:
    return (
        summary["generated_text"],
        summary["estimated_saved_percent"],
        summary["expensive_model_calls"],
    )


def run_task_evaluation(
    dataset_path: str | Path = "data/eval_tasks.jsonl",
    cheap_model_name: str = AdaptiveGenerationConfig.cheap_model_name,
    expensive_model_name: str = AdaptiveGenerationConfig.expensive_model_name,
    max_new_tokens: int = 80,
    temperature: float = 0.7,
    device: str = "auto",
    torch_dtype: str = "auto",
    prompt_format: str = "auto",
    models=None,
) -> tuple[list[dict], list[dict]]:
    tasks = load_eval_tasks(dataset_path)

    if models is None:
        config = AdaptiveGenerationConfig(
            cheap_model_name=cheap_model_name,
            expensive_model_name=expensive_model_name,
            device=device,
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
    if cheap_runtime != expensive_runtime:
        raise ValueError(
            "cheap_model and expensive_model must use the same device/dtype. "
            f"cheap={cheap_runtime}, expensive={expensive_runtime}"
        )

    model_metadata = {
        "cheap_model_name": cheap_model_name,
        "expensive_model_name": expensive_model_name,
        "device": cheap_runtime["device"],
        "torch_dtype": cheap_runtime["torch_dtype"],
        **prompt_format_metadata(tokenizer, prompt_format),
    }
    cheap_tokenizer = get_cheap_tokenizer(tokenizer)
    expensive_tokenizer = get_expensive_tokenizer(tokenizer)
    rows = []

    for task in tasks:
        prompt = task["prompt"]
        mode_summaries = {}

        expensive_text, expensive_tokens = generate_greedy_with_model(
            prompt=prompt,
            model=expensive_model,
            tokenizer=expensive_tokenizer,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            prompt_format=prompt_format,
        )
        evaluation = evaluate_task(task, expensive_text)
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
            )
        )

        cheap_text, cheap_tokens = generate_greedy_with_model(
            prompt=prompt,
            model=cheap_model,
            tokenizer=cheap_tokenizer,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            prompt_format=prompt_format,
        )
        cheap_saved = estimated_saved_percent(
            total_generated_tokens=cheap_tokens,
            cheap_calls=cheap_tokens,
            expensive_calls=0,
        )
        evaluation = evaluate_task(task, cheap_text)
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
            )
        )

        for mode in (
            "adaptive_calibrated",
            "adaptive_guarded_v3",
            "speculative_adaptive",
        ):
            summary = generate_with_mode(
                prompt=prompt,
                mode=mode,
                cheap_model=cheap_model,
                expensive_model=expensive_model,
                tokenizer=tokenizer,
                device=device,
                cheap_model_name=cheap_model_name,
                expensive_model_name=expensive_model_name,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                prompt_format=prompt_format,
            )
            mode_summaries[mode] = summary
            generated_text, saved, expensive_calls = _generation_from_summary(
                summary
            )
            evaluation = evaluate_task(task, generated_text)
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
                )
            )

        prompt_type = classify_prompt(prompt)
        selected_mode = choose_mode(prompt_type, prompt)
        hybrid_summary = mode_summaries[selected_mode]
        generated_text, saved, expensive_calls = _generation_from_summary(
            hybrid_summary
        )
        evaluation = evaluate_task(task, generated_text)
        rows.append(
            _build_row(
                task=task,
                mode="hybrid",
                selected_mode=selected_mode,
                generated_text=generated_text,
                evaluation=evaluation,
                estimated_saved=saved,
                expensive_model_calls=expensive_calls,
                model_metadata=model_metadata,
                max_new_tokens=max_new_tokens,
            )
        )

    return rows, summarize_task_evaluation(rows)


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
        count = len(group)
        passed = sum(1 for row in group if row["passed"])
        score_sum = sum(float(row["score"]) for row in group)
        saved_sum = sum(float(row["estimated_saved_percent"]) for row in group)
        calls_sum = sum(float(row["expensive_model_calls"]) for row in group)

        summary_rows.append(
            {
                "category": category,
                "mode": mode,
                "count": count,
                "pass_rate": passed / count if count else 0.0,
                "avg_score": score_sum / count if count else 0.0,
                "avg_estimated_saved_percent": saved_sum / count if count else 0.0,
                "avg_expensive_model_calls": calls_sum / count if count else 0.0,
            }
        )

    return summary_rows


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


def save_task_evaluation_outputs(
    rows: list[dict],
    summary_rows: list[dict],
    output_dir: str | Path = "results",
) -> tuple[Path, Path]:
    output_path = Path(output_dir)
    detailed_csv = output_path / "task_evaluation.csv"
    summary_csv = output_path / "task_evaluation_summary.csv"

    save_csv(rows, str(detailed_csv))
    save_csv(summary_rows, str(summary_csv))

    return detailed_csv, summary_csv
