"""
Walkthrough of what the map can do that a static index can't.

    python research/synthetic_demo.py
"""
import time

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mapnet import MapNet
from mapnet.synthetic import build_company, make_questions


def section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def show(res, n=3):
    print(f"  route: {' -> '.join(', '.join(level) for level in res.trace)}")
    print(f"  searched {len(res.sources_searched)} sources, {res.cost:,} comparisons"
          f"{' (escalated)' if res.escalated else ''}")
    for h in res.hits[:n]:
        print(f"    {h.score:.3f}  {h.source:35} {h.text[:60]}")


sources, facts = build_company(sources_per_topic=8, noise_ratio=0.5)
m = MapNet()
for s in sources:
    m.register_source(**s)
total = sum(len(s["docs"]) for s in sources)

section(f"1. The map: {len(sources)} sources, {total:,} documents")
m.show()

section("2. A query walks the map instead of scanning everything")
q, gold = make_questions(facts, 1, seed=7)[0]
print(f"  query: {q!r}   (answer: {gold})")
show(m.search(q))
print(f"  flat search would score all {total:,} documents")

section("3. Live update: a new system joins, no re-indexing")
t0 = time.perf_counter()
m.register_source("statuspage", [
    ("eng.statuspage.0", "zephyr999: outage incident failover latency severity (ref zephyr999)"),
    ("eng.statuspage.1", "nimbus998: postmortem rollback alert mitigation (ref nimbus998)"),
], path=["engineering", "incidents"])
print(f"  registered in {(time.perf_counter() - t0) * 1000:.1f} ms")
show(m.search("what is the status of zephyr999 outage"))
t0 = time.perf_counter()
m.add_documents("engineering/incidents/statuspage",
                [("eng.statuspage.2", "zephyr999 resolved: oncall downtime closed (ref zephyr999)")])
print(f"  pushed one more document in {(time.perf_counter() - t0) * 1000:.1f} ms; asking again:")
show(m.search("zephyr999 resolved downtime"))

section("4. Self-placement: a source with no path finds its own spot")
node = m.register_source("vendor-portal", [
    (f"vp.{i}", f"vendor{i}: invoice remittance billing overdue payment (ref vendor{i})")
    for i in range(10)])
print(f"  'vendor-portal' placed itself at: {node.path}")

section("5. Permissions: restricted branches are never contacted")
q = "what is the status of the payslip salary bonus"
roles = {"engineering"}
r = m.search(q, roles=roles)
f = m.flat_search(q, roles=roles)
leaked = [h for h in r.hits if "payroll" in h.source]
print(f"  user roles: {roles}; hr/payroll is restricted to 'hr'")
print(f"  map:  {r.cost:,} comparisons, payroll docs touched: 0, payroll hits returned: {len(leaked)}")
print(f"  flat: {f.cost:,} comparisons, of which {f.restricted_docs_scanned:,} were restricted docs "
      f"scored and then filtered out")
print("  (in a real deployment, 'scored then filtered' means the data was read by the wrong caller)")
r = m.search(q, roles={"hr"})
print(f"  same query as an 'hr' user -> top source: {r.hits[0].source}")

section("6. Feedback: the map learns how people actually ask")
m2 = MapNet(name_weight=0.0)  # summaries only, to isolate the effect
for s in sources:
    m2.register_source(**s)
test = make_questions(facts, 200, seed=1, slang=True)
train = make_questions(facts, 2000, seed=2, slang=True, exclude=[g for _, g in test])


def acc(mp):
    return sum(mp.search(q).hits[0].doc_id == g for q, g in test) / len(test)


print(f"  slang questions, e.g. {test[0][0]!r}")
print(f"  right answer ranked first, before feedback: {acc(m2):.0%}")
for q, g in train:
    m2.feedback(q, g)
print(f"  right answer ranked first, after learning from {len(train):,} answered questions: {acc(m2):.0%}")
