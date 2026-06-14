"""
LLM auto-grader: compares each result's actual answer against the verified
expected answer and writes a grade back into the results file.

Usage:
  python grade.py --set results_v3.json
  python grade.py --set results_holdout.json
"""
import os
import sys
import json
import time
from pathlib import Path
from dotenv import load_dotenv

import anthropic

load_dotenv()

GRADER_MODEL = "claude-sonnet-4-6"

GRADE_SYSTEM = (
    "You are a strict but fair grader for a financial question-answering system. "
    "You compare the system's ANSWER against a verified EXPECTED answer.\n\n"
    "For ANSWERABLE questions, grade:\n"
    "  correct  - the key fact(s)/figure(s) match the expected answer (small rounding or "
    "wording differences are fine; a different but genuinely valid figure from the same "
    "filing — e.g. cash paid for buybacks vs value of shares repurchased — counts as correct)\n"
    "  partial  - partially right, or right direction but missing or misstating a key number\n"
    "  wrong    - the key fact is wrong, or it refused/failed to answer an answerable question\n\n"
    "For UNANSWERABLE questions (expected answer starts with REFUSE), grade:\n"
    "  correct_refuse - the system declined / said it lacks the data / asked to add the company\n"
    "  wrong_refuse   - the system fabricated an answer instead of refusing\n\n"
    "Reply with ONLY a JSON object: {\"grade\": \"<grade>\", \"reason\": \"<one short sentence>\"}"
)


def _resolve(name_default: str) -> Path:
    for i, arg in enumerate(sys.argv):
        if arg == "--set" and i + 1 < len(sys.argv):
            return Path(__file__).parent / sys.argv[i + 1]
    return Path(__file__).parent / name_default


def grade_file(path: Path) -> None:
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    results = json.loads(path.read_text(encoding="utf-8"))

    for r in results:
        prompt = (
            f"QUESTION: {r['question']}\n\n"
            f"EXPECTED: {r['expected']}\n\n"
            f"SYSTEM ANSWER: {r['actual']}\n\n"
            "Grade the system answer."
        )
        for attempt in range(3):
            try:
                resp = client.messages.create(
                    model=GRADER_MODEL,
                    max_tokens=200,
                    system=GRADE_SYSTEM,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = resp.content[0].text.strip()
                if raw.startswith("```"):
                    raw = raw.split("```", 2)[1].lstrip("json").strip()
                verdict = json.loads(raw)
                r["grade"] = verdict["grade"]
                r["notes"] = verdict.get("reason", "")
                break
            except Exception as exc:
                if attempt == 2:
                    r["grade"] = "ungraded"
                    r["notes"] = f"grader error: {exc}"
                time.sleep(2)
        print(f"[{r['id']}] {r['grade']}: {r['notes']}")

    path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nGraded results written to {path}")


if __name__ == "__main__":
    grade_file(_resolve("results_v3.json"))
