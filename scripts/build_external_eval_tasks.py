import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path


SOURCE_ROOT = Path("data/external_sources")


def read_records(path: Path) -> list[dict]:
    if path.suffix.lower() == ".txt":
        return read_logiqa_original_txt(path)

    if path.suffix == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                stripped = line.strip()
                if stripped:
                    records.append(json.loads(stripped))
        return records

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("data", "examples", "test", "validation", "train"):
            value = data.get(key)
            if isinstance(value, list):
                return value

    raise ValueError(f"Unsupported JSON structure in {path}")


def read_logiqa_original_txt(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    records = []
    index = 0

    while index < len(lines):
        if not lines[index].strip():
            chunk = lines[index : index + 8]
            index += 8
            if len(chunk) < 8:
                break
            answer = chunk[1]
            passage = chunk[2]
            question = chunk[3]
            options = chunk[4:8]
        else:
            chunk = lines[index : index + 7]
            index += 7
            if len(chunk) < 7:
                break
            answer = chunk[0]
            passage = chunk[1]
            question = chunk[2]
            options = chunk[3:7]

        if not str(answer).strip() or len(options) < 4:
            continue

        records.append(
            {
                "answer": answer.strip(),
                "passage": passage.strip(),
                "question": question.strip(),
                "options": [strip_option_label(option) for option in options],
            }
        )

    return records


def source_missing_message(source: str, candidates: list[Path]) -> str:
    candidate_text = "\n".join(f"  - {path}" for path in candidates)
    return (
        f"Could not find local source files for '{source}'.\n"
        "Place/download the benchmark files under data/external_sources/.\n"
        "Expected one of:\n"
        f"{candidate_text}\n"
        "Examples:\n"
        "  GSM8K: data/external_sources/gsm8k/test.jsonl or test.json\n"
        "  MBPP : data/external_sources/mbpp/mbpp.jsonl or sanitized-mbpp.json\n"
        "  LogiQA: data/external_sources/logiqa/test.jsonl, test.json, or Test.txt"
    )


def find_source_file(source: str, kind: str) -> Path:
    if source == "sample":
        candidates = [SOURCE_ROOT / "sample" / f"{kind}.jsonl"]
    elif kind == "gsm8k":
        candidates = [
            SOURCE_ROOT / source / "test.jsonl",
            SOURCE_ROOT / source / "test.json",
        ]
    elif kind == "mbpp":
        candidates = [
            SOURCE_ROOT / source / "mbpp.jsonl",
            SOURCE_ROOT / source / "sanitized-mbpp.json",
            SOURCE_ROOT / source / "sanitized-mbpp.jsonl",
        ]
    elif kind == "logiqa":
        candidates = [
            SOURCE_ROOT / source / "test.jsonl",
            SOURCE_ROOT / source / "test.json",
            SOURCE_ROOT / source / "Test.txt",
        ]
    else:
        raise ValueError(f"Unsupported source kind: {kind}")

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(source_missing_message(source, candidates))


def strip_option_label(option: str) -> str:
    return re.sub(r"^\s*[A-Da-d][\.\)\]、:]\s*", "", str(option)).strip()


def normalize_number_answer(answer: str) -> str:
    answer = str(answer).strip()
    answer = answer.replace(",", "")
    answer = answer.replace("$", "")
    answer = re.sub(r"\s+", "", answer)
    answer = answer.rstrip(".")
    return answer


def parse_gsm8k_answer(answer_field: str) -> str:
    text = str(answer_field)
    match = re.search(r"####\s*([^\n]+)", text)
    if match:
        return normalize_number_answer(match.group(1))

    numbers = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text)
    if numbers:
        return normalize_number_answer(numbers[-1])

    return normalize_number_answer(text)


def unique(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def convert_gsm8k(record: dict, index: int) -> dict | None:
    question = record.get("question") or record.get("prompt") or record.get("problem")
    answer_field = record.get("answer") or record.get("target") or record.get("final_answer")
    if not question or answer_field is None:
        return None

    final = parse_gsm8k_answer(answer_field)
    acceptable = unique([final, final.replace(",", "")])

    return {
        "id": f"gsm8k_{index:06d}",
        "category": "math",
        "source": "gsm8k",
        "difficulty": "medium",
        "prompt": f"{question.strip()}\n\nAnswer only the final number.",
        "expected_answer": final,
        "acceptable_answers": acceptable,
        "answer_type": "number",
    }


def infer_function_name(record: dict, tests: list[str]) -> str:
    for test in tests:
        match = re.search(r"assert\s+([A-Za-z_]\w*)\s*\(", test)
        if match:
            return match.group(1)

    code = record.get("code") or record.get("canonical_solution") or ""
    match = re.search(r"\bdef\s+([A-Za-z_]\w*)\s*\(", str(code))
    if match:
        return match.group(1)

    return str(record.get("function_name") or "").strip()


def normalize_mbpp_tests(raw_tests) -> list[dict]:
    if isinstance(raw_tests, str):
        raw_tests = [line.strip() for line in raw_tests.splitlines() if line.strip()]

    tests = []
    for test in raw_tests or []:
        if isinstance(test, dict):
            if "call" in test and "expected" in test:
                tests.append(test)
            elif "assert" in test:
                tests.append({"assert": str(test["assert"]).strip()})
            elif "raw_assert" in test:
                tests.append({"assert": str(test["raw_assert"]).strip()})
            continue

        text = str(test).strip()
        if text.startswith("assert "):
            tests.append({"assert": text})

    return tests


def mbpp_difficulty(text: str, tests: list[dict]) -> str:
    combined = text.lower() + " " + " ".join(str(test) for test in tests).lower()
    hard_terms = (
        "list",
        "string",
        "dictionary",
        "dict",
        "sort",
        "duplicate",
        "palindrome",
        "matrix",
        "multiple",
        "condition",
        "recursive",
    )
    easy_terms = ("sum", "add", "square", "even", "odd", "maximum", "minimum")
    if any(term in combined for term in hard_terms) or len(tests) >= 4:
        return "hard"
    if any(term in combined for term in easy_terms) and len(tests) <= 2:
        return "easy"
    return "medium"


def convert_mbpp(record: dict, index: int) -> dict | None:
    text = (
        record.get("text")
        or record.get("prompt")
        or record.get("task")
        or record.get("description")
    )
    raw_tests = record.get("test_list") or record.get("tests") or []
    tests = normalize_mbpp_tests(raw_tests)
    function_name = infer_function_name(record, [test.get("assert", "") for test in tests])
    if not text or not tests or not function_name:
        return None

    prompt = (
        "Write a Python function for the task below. "
        f"The function must be named {function_name}. "
        "Return only Python code.\n\n"
        f"Task: {str(text).strip()}"
    )

    return {
        "id": f"mbpp_{index:06d}",
        "category": "code",
        "source": "mbpp",
        "difficulty": mbpp_difficulty(str(text), tests),
        "prompt": prompt,
        "function_name": function_name,
        "tests": tests,
    }


def normalize_options(record: dict) -> list[str]:
    options = record.get("options") or record.get("choices") or record.get("answers")
    if isinstance(options, dict):
        result = []
        for key in ("A", "B", "C", "D", "a", "b", "c", "d", "0", "1", "2", "3"):
            if key in options:
                result.append(str(options[key]))
        if len(result) >= 4:
            return result[:4]
    if isinstance(options, list):
        return [strip_option_label(option) for option in options[:4]]

    result = []
    for key in ("option_0", "option_1", "option_2", "option_3"):
        if key in record:
            result.append(strip_option_label(record[key]))
    return result


def normalize_choice_answer(answer, options: list[str]) -> str:
    if isinstance(answer, int):
        return "ABCD"[answer] if 0 <= answer < 4 else ""

    text = str(answer).strip()
    if re.fullmatch(r"[A-Da-d]", text):
        return text.upper()
    if re.fullmatch(r"[0-3]", text):
        return "ABCD"[int(text)]

    for index, option in enumerate(options):
        if text.lower() == option.lower():
            return "ABCD"[index]

    match = re.search(r"\b([A-Da-d])\b", text)
    return match.group(1).upper() if match else ""


def convert_logiqa(record: dict, index: int) -> dict | None:
    passage = record.get("passage") or record.get("context") or record.get("article")
    question = record.get("question") or record.get("query") or record.get("prompt")
    options = normalize_options(record)
    answer = (
        record.get("answer")
        if "answer" in record
        else record.get("label", record.get("correct_answer"))
    )
    letter = normalize_choice_answer(answer, options)
    if not question or len(options) < 4 or not letter:
        return None

    option_lines = [f"{letter_name}. {option}" for letter_name, option in zip("ABCD", options)]
    context = f"Passage: {passage.strip()}\n\n" if passage else ""
    prompt = (
        f"{context}Question: {str(question).strip()}\n"
        + "\n".join(option_lines)
        + "\n\nAnswer only with one letter: A, B, C, or D."
    )

    return {
        "id": f"logiqa_{index:06d}",
        "category": "logic",
        "source": "logiqa",
        "difficulty": "medium",
        "prompt": prompt,
        "expected_answer": letter,
        "acceptable_answers": [letter],
        "answer_type": "choice",
    }


def build_source_tasks(source: str, kind: str, limit: int | None, seed: int) -> list[dict]:
    if limit is not None and limit <= 0:
        return []

    path = find_source_file(source, kind)
    records = read_records(path)
    rng = random.Random(seed)
    rng.shuffle(records)

    converters = {
        "gsm8k": convert_gsm8k,
        "mbpp": convert_mbpp,
        "logiqa": convert_logiqa,
    }
    converter = converters[kind]

    tasks = []
    skipped = 0
    for record in records:
        task = converter(record, len(tasks) + 1)
        if task is None:
            skipped += 1
            continue
        tasks.append(task)
        if limit is not None and len(tasks) >= limit:
            break

    print(
        f"{kind:<6} source={source:<8} loaded={len(records)} "
        f"converted={len(tasks)} skipped={skipped} file={path}"
    )
    return tasks


def write_jsonl(tasks: list[dict], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for task in tasks:
            file.write(json.dumps(task, ensure_ascii=False) + "\n")


def print_counts(tasks: list[dict]):
    print()
    print("External eval task counts")
    print("=" * 72)
    counter = Counter(
        (task["category"], task.get("source", ""), task.get("difficulty", ""))
        for task in tasks
    )
    for (category, source, difficulty), count in sorted(counter.items()):
        print(f"{category:<8} | {source:<8} | {difficulty:<6} | {count:>5}")
    print("=" * 72)
    print(f"total: {len(tasks)}")


def main():
    parser = argparse.ArgumentParser(
        description="Build normalized external task-evaluation JSONL datasets."
    )
    parser.add_argument("--output", type=str, default="data/external_eval_tasks.jsonl")
    parser.add_argument("--math-source", type=str, default="gsm8k")
    parser.add_argument("--code-source", type=str, default="mbpp")
    parser.add_argument("--logic-source", type=str, default="logiqa")
    parser.add_argument("--math-limit", type=int, default=None)
    parser.add_argument("--code-limit", type=int, default=None)
    parser.add_argument("--logic-limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    any_limit_set = any(
        limit is not None
        for limit in (args.math_limit, args.code_limit, args.logic_limit)
    )
    tasks = []
    if not any_limit_set or args.math_limit is not None:
        tasks.extend(
            build_source_tasks(
                source=args.math_source,
                kind="gsm8k",
                limit=args.math_limit,
                seed=args.seed,
            )
        )
    if not any_limit_set or args.code_limit is not None:
        tasks.extend(
            build_source_tasks(
                source=args.code_source,
                kind="mbpp",
                limit=args.code_limit,
                seed=args.seed,
            )
        )
    if not any_limit_set or args.logic_limit is not None:
        tasks.extend(
            build_source_tasks(
                source=args.logic_source,
                kind="logiqa",
                limit=args.logic_limit,
                seed=args.seed,
            )
        )

    output_path = Path(args.output)
    write_jsonl(tasks, output_path)
    print_counts(tasks)
    print(f"wrote: {output_path}")


if __name__ == "__main__":
    main()
