import requests

BASE = "http://127.0.0.1:8000"

# Each case: a question, and a fact the correct answer MUST contain.
CASES = [
    {"q": "How many repositories do I have in total?", "must_contain": "13"},
    {"q": "Which of my repositories are written in Python?",
        "must_contain": "expense-tracker"},
    {"q": "List my TypeScript repositories.", "must_contain": "portfolio"},
    {"q": "Which repos are written in CSS?", "must_contain": "f-backroads-app"},
]

passed = 0
for case in CASES:
    answer = requests.get(
        f"{BASE}/agent", params={"q": case["q"]}).json().get("answer", "")
    ok = case["must_contain"].lower() in answer.lower()
    print(f"[{'PASS' if ok else 'FAIL'}] {case['q']}")
    print(f"        expected to contain: {case['must_contain']!r}")
    print(f"        got: {answer}\n")
    passed += ok

print(f"=== {passed}/{len(CASES)} checks passed ===")
