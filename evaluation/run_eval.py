import json
import time
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.rag import RAGPipeline

# Test set can be overridden on the command line, e.g.:
#   python run_eval.py --set test_set_holdout.json
#   python run_eval.py --set test_set_holdout.json --score
def _resolve_test_set() -> Path:
    for i, arg in enumerate(sys.argv):
        if arg == "--set" and i + 1 < len(sys.argv):
            return Path(__file__).parent / sys.argv[i + 1]
    return Path(__file__).parent / "test_set_v3.json"


TEST_SET     = _resolve_test_set()
RESULTS_FILE = Path(__file__).parent / (TEST_SET.stem.replace("test_set", "results") + ".json")


def _save(results: list) -> None:
    RESULTS_FILE.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def run_evaluation() -> None:
    questions = json.loads(TEST_SET.read_text(encoding="utf-8"))
    pipeline = RAGPipeline()

    # Resume from existing results if any
    if RESULTS_FILE.exists():
        existing = json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
        done_ids = {r["id"] for r in existing}
        results = existing
        print(f"Resuming — {len(done_ids)} questions already done.")
    else:
        done_ids = set()
        results = []

    for q in questions:
        qid = q["id"]
        if qid in done_ids:
            print(f"[{qid}] skipped (already done)")
            continue

        category = q["category"]
        print(f"\n[{qid}] {q['question'][:80]}...")

        start = time.time()
        try:
            result = pipeline.query(q["question"], k=10)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            result_entry = {
                "id": qid,
                "category": category,
                "question": q["question"],
                "expected": q["expected_answer"],
                "actual": f"ERROR: {exc}",
                "sources": [],
                "rewritten_query": "",
                "is_grounded": False,
                "elapsed_s": round(time.time() - start, 1),
                "grade": None,
                "notes": "pipeline error",
            }
            results.append(result_entry)
            _save(results)
            continue

        elapsed = round(time.time() - start, 1)
        results.append({
            "id": qid,
            "category": category,
            "question": q["question"],
            "expected": q["expected_answer"],
            "actual": result.answer,
            "sources": [s["source"] for s in result.sources],
            "rewritten_query": result.rewritten_query,
            "is_grounded": result.is_grounded,
            "elapsed_s": elapsed,
            "grade": None,
            "notes": "",
        })
        _save(results)
        print(f"  Answer: {result.answer[:120]}...")
        print(f"  Sources: {[s['source'] for s in result.sources[:2]]}")
        print(f"  Grounded: {result.is_grounded} | {elapsed}s")

    print(f"\n\nResults saved to {RESULTS_FILE}")
    print("Next: open the results file and fill in the 'grade' field for each question.")
    print("Grades: correct | partial | wrong | correct_refuse | wrong_refuse")


def score_results() -> None:
    if not RESULTS_FILE.exists():
        print("Run run_evaluation() first.")
        return

    results = json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
    graded = [r for r in results if r["grade"]]

    if not graded:
        print("No grades found yet. Fill in the 'grade' field in results.json first.")
        return

    answerable   = [r for r in graded if r["category"] != "unanswerable"]
    unanswerable = [r for r in graded if r["category"] == "unanswerable"]

    correct = sum(1 for r in answerable if r["grade"] == "correct")
    partial = sum(1 for r in answerable if r["grade"] == "partial")
    wrong   = sum(1 for r in answerable if r["grade"] == "wrong")

    correct_refuse = sum(1 for r in unanswerable if r["grade"] == "correct_refuse")
    wrong_refuse   = sum(1 for r in unanswerable if r["grade"] == "wrong_refuse")
    grounded_answerable = sum(1 for r in answerable if r["is_grounded"])

    print("\n========== EVALUATION RESULTS ==========")
    print(f"\nAnswerable questions ({len(answerable)} total):")
    print(f"  Correct:  {correct}/{len(answerable)}  ({100*correct/len(answerable):.0f}%)")
    print(f"  Partial:  {partial}/{len(answerable)}")
    print(f"  Wrong:    {wrong}/{len(answerable)}")
    print(f"  Grounded: {grounded_answerable}/{len(answerable)}  ({100*grounded_answerable/len(answerable):.0f}%)")

    # Per-category breakdown
    for cat in ("factual", "analytical", "cross_company"):
        cat_rows = [r for r in answerable if r["category"] == cat]
        if not cat_rows:
            continue
        cat_correct = sum(1 for r in cat_rows if r["grade"] == "correct")
        cat_partial = sum(1 for r in cat_rows if r["grade"] == "partial")
        print(f"\n  {cat} ({len(cat_rows)}):")
        print(f"    Correct: {cat_correct}  Partial: {cat_partial}  Wrong: {len(cat_rows)-cat_correct-cat_partial}")

    if unanswerable:
        halluc_rate = wrong_refuse / len(unanswerable)
        print(f"\nUnanswerable questions ({len(unanswerable)} total):")
        print(f"  Correctly refused: {correct_refuse}/{len(unanswerable)}")
        print(f"  Hallucinated:      {wrong_refuse}/{len(unanswerable)}")
        print(f"  Hallucination rate: {halluc_rate*100:.0f}%")

    # Wrong / partial answers listed for easy inspection
    problem_rows = [r for r in answerable if r["grade"] in ("wrong", "partial")]
    if problem_rows:
        print("\n--- Questions needing attention ---")
        for r in problem_rows:
            print(f"  [{r['id']}] {r['grade'].upper()}: {r['question'][:80]}")

    print(f"\nTotal graded: {len(graded)}/{len(results)}")
    print("=========================================")


if __name__ == "__main__":
    if "--score" in sys.argv:
        score_results()
    else:
        run_evaluation()
