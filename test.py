"""
Manual smoke test: runs 20 questions through the LangGraph pipeline and
saves the results to test_results.json.

Covers: refusals (off-topic, other companies), metric questions, comparisons,
narrative questions, and edge cases (no year, future year, relative year).
Questions are intentionally in Russian and English to check bilingual input.

Run: python test.py  (uses ~2 LLM calls per question, mind the free-tier limits)
"""
import json
import time
from pathlib import Path
from graph import app

BASE_DIR = Path(__file__).resolve().parent
RESULTS_PATH = BASE_DIR / "test_results.json"
# (question, expected question_type or None if it doesn't matter,
#  should metrics be returned: True/False/None)
CASES = [
    ("Привет, как дела?",                                  "other",      False),  # greeting
    ("Какая выручка у Tesla в 2023?",                      "other",      False),  # Tesla revenue 2023 (unsupported company)
    ("Расскажи анекдот",                                   "other",      False),  # tell a joke
    ("Какие риски у Google?",                              "other",      False),  # Google risks (unsupported company)
    ("Какая выручка у Apple в 2023?",                      "metric",     True),   # Apple revenue 2023
    ("What was Microsoft's net income in 2022?",           "metric",     True),
    ("Чистая прибыль Amazon за 2021 и 2023",               "metric",     True),   # Amazon net income 2021 and 2023
    ("Какая выручка у Apple?",                             None,         False),  # Apple revenue, no year given
    ("Какие основные риски у Amazon в 2023?",              "narrative",  False),  # Amazon main risks 2023 (text is FY2025 only)
    ("Why did Apple's revenue change in 2022?",            "narrative",  None),
    ("Расскажи про стратегию Microsoft",                   "narrative",  False),  # Microsoft strategy
    ("Какие риски указаны в отчётах?",                     "narrative",  False),  # risks in the reports (no company)
    ("Сравни выручку Apple и Microsoft в 2023",            "comparison", True),   # compare Apple and Microsoft revenue 2023
    ("Чем отличаются риски Amazon и Apple?",               "comparison", False),  # how Amazon and Apple risks differ
    ("Что общего у всех трёх компаний в плане конкуренции?", "comparison", False),  # what all three have in common on competition
    ("Какая выручка у Apple в 2035?",                      None,         False),  # Apple revenue 2035 (no such data)
    ("Какая прибыль у Apple в прошлом году?",              None,         False),  # Apple profit last year (relative year)
    ("Сколько сотрудников у Amazon в 2023?",               None,         False),  # Amazon headcount (metric not supported)
    ("iPhone vs Mac: что приносит больше выручки?",        None,         None),   # iPhone vs Mac revenue
    ("?",                                                  "other",      False),  # garbage input
]


def run_one(question, retries=3):
    for attempt in range(retries):
        try:
            return app.invoke({"question": question})
        except Exception as e:
            if "429" in str(e) and attempt < retries - 1:
                print(f"   429, waiting 30s (attempt {attempt + 1}/{retries})")
                time.sleep(30)
                continue
            if "per-day" in str(e):
                print("Daily limit reached, no point continuing")
                raise SystemExit
            raise


results = []

for i, (q, exp_type, exp_metrics) in enumerate(CASES, 1):
    print(f"\n=== {i}/{len(CASES)}: {q}")
    row = {"question": q}
    try:
        res = run_one(q)
        router = json.loads(res["router_ans"])
        got_type = router["question_type"]
        has_metrics = bool(res.get("metric_ans"))
        answer = res.get("RAG_ans", "")

        problems = []
        if exp_type and got_type != exp_type:
            problems.append(f"type: expected {exp_type}, got {got_type}")
        if exp_metrics is not None and has_metrics != exp_metrics:
            problems.append(f"metrics: expected {exp_metrics}, got {has_metrics}")
        if not answer.strip():
            problems.append("empty answer")

        row.update(
            status="OK" if not problems else "CHECK",
            problems=problems,
            router=router,
            metric_ans=res.get("metric_ans"),
            answer=answer,
        )
        print(f"   router: type={got_type}, companies={router['companies']}, "
              f"year={router['year']}, metric={router['metric']}")
        print(f"   metrics: {res.get('metric_ans')!r}")
        print(f"   answer: {answer}")
        print(f"   [{row['status']}] {'; '.join(problems)}")
    except Exception as e:
        row.update(status="ERROR", problems=[str(e)[:200]])
        print(f"   [ERROR] {e}")

    results.append(row)
    # "other" costs 1 LLM call, the rest cost 2: stay under the 15 requests/min limit
    time.sleep(4 if row.get("router", {}).get("question_type") == "other" else 9)

# summary
print("\n" + "=" * 60)
for status in ("OK", "CHECK", "ERROR"):
    n = sum(r["status"] == status for r in results)
    print(f"{status}: {n}")
for r in results:
    if r["status"] != "OK":
        print(f"- [{r['status']}] {r['question']}: {'; '.join(r['problems'])}")

with open(RESULTS_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"\nAll saved to {RESULTS_PATH.name}")