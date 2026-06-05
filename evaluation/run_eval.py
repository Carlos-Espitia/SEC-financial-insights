import json
import time
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.rag import RAGPipeline

TEST_SET   = Path(__file__).parent / "test_set_v2.json"
RESULTS_FILE = Path(__file__).parent / "results.json"


def run_evaluation() -> None:
    questions = json.loads(TEST_SET.read_text(encoding="utf-8"))
    pipeline = RAGPipeline()

    results = []
    for q in questions:
        qid = q["id"]
        category = q["category"]
        print(f"\n[{qid}] {q['question'][:80]}...")

        start = time.time()
        # No explicit filters — intent detection handles ticker, period, form_type, sections
        result = pipeline.query(q["question"], k=8)
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
        print(f"  Answer: {result.answer[:120]}...")
        print(f"  Sources: {[s['source'] for s in result.sources[:2]]}")
        print(f"  Grounded: {result.is_grounded} | {elapsed}s")

    RESULTS_FILE.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n\nResults saved to {RESULTS_FILE}")
    print("Next: open results.json and fill in the 'grade' field for each question.")
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

    if unanswerable:
        halluc_rate = wrong_refuse / len(unanswerable)
        print(f"\nUnanswerable questions ({len(unanswerable)} total):")
        print(f"  Correctly refused: {correct_refuse}/{len(unanswerable)}")
        print(f"  Hallucinated:      {wrong_refuse}/{len(unanswerable)}")
        print(f"  Hallucination rate: {halluc_rate*100:.0f}%")

    print(f"\nTotal graded: {len(graded)}/{len(results)}")
    print("=========================================")


if __name__ == "__main__":
    if "--score" in sys.argv:
        score_results()
    else:
        run_evaluation()
