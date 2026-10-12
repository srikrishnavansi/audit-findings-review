"""Eval harness. Runs the graph over the labelled notes, then scores:
- article and risk accuracy against the hand labels,
- judge agreement: the judge's verdict against known truth, on the real drafts (is the citation right?)
  and on four broken copies of every draft (near-miss article, invented figure, invented claim,
  vague recommendation).

    python evals.py      # prints the tables, writes out/run.json (review queue) and out/eval.json
"""
import json
from datetime import datetime, timezone

from pipeline import HERE, flags, judge, llm_from_env, load_notes, mode_name, run, to_run

LABELS = {k: v for k, v in json.loads((HERE / "data/labels.json").read_text()).items() if k != "about"}
INVENTED_FIGURE = " Vendor Northwind Data confirmed 1,240 affected records on 03.07."
INVENTED_CLAIM = " The same gap was raised in last year's audit and accepted by management."
VAGUE = "Consider reviewing this area as appropriate."
BROKEN = (("citation", "near-miss article", lambda d, alt: {**d, "article": alt}),
          ("faithful", "invented figure", lambda d, alt: {**d, "evidence": d["evidence"] + INVENTED_FIGURE}),
          ("faithful", "invented claim", lambda d, alt: {**d, "evidence": d["evidence"] + INVENTED_CLAIM}),
          ("actionable", "vague recommendation", lambda d, alt: {**d, "recommendation": VAGUE}))
KINDS = (("citation", "real draft"),) + tuple((check, kind) for check, kind, _ in BROKEN)


def evaluate(results, llm=None):
    ids = [r["note"]["id"] for r in results]
    rows, cases = [], []
    for i, r in enumerate(results):
        nid, d, lab, note = ids[i], r["draft"], LABELS[ids[i]], r["note"]["text"]
        rows.append({"id": nid, "article_expected": lab["article"], "article": d["article"],
                     "risk_expected": lab["risk"], "risk": d["risk"], "flags": flags(r["judge"])})
        cases.append({"id": nid, "kind": "real draft", "check": "citation",
                      "expected_ok": d["article"] == lab["article"], "judge_ok": r["judge"]["citation"]["ok"]})
        # near miss: the best-ranked retrieved article that is not the labelled one
        alt = next(a for a, _ in r["candidates"] if a != lab["article"])
        for check, kind, broken in BROKEN:
            cases.append({"id": nid, "kind": kind, "check": check, "expected_ok": False,
                          "judge_ok": judge(note, broken(d, alt), llm)[check]["ok"]})
    n = len(rows)
    summary = {
        "mode": mode_name(llm), "notes": n,
        "article_correct": sum(r["article"] == r["article_expected"] for r in rows),
        "risk_correct": sum(r["risk"] == r["risk_expected"] for r in rows),
        "judge_agree": sum(c["expected_ok"] == c["judge_ok"] for c in cases), "judge_cases": len(cases),
        "by_kind": {kind: sum(c["expected_ok"] == c["judge_ok"] for c in cases if c["kind"] == kind)
                    for _, kind in KINDS},
        "real_drafts_flagged": sum(bool(r["flags"]) for r in rows),
    }
    if llm:
        summary.update(llm_calls=llm.calls, llm_usd=round(llm.spent, 4))
    return {"summary": summary, "rows": rows, "cases": cases}


def table(headers, rows):
    w = [max(len(str(x)) for x in col) for col in zip(headers, *rows)]
    line = lambda r: "  ".join(str(x).ljust(n) for x, n in zip(r, w)).rstrip()
    return "\n".join([line(headers), line(["-" * n for n in w])] + [line(r) for r in rows])


def pct(a, b):
    return f"{a}/{b}  {100 * a / b:.0f}%"


def report(ev):
    s, n = ev["summary"], ev["summary"]["notes"]
    per_note = table(["note", "expected article", "drafted", "", "expected risk", "drafted", "", "judge flags"],
                     [[r["id"], r["article_expected"], r["article"], "ok" if r["article"] == r["article_expected"] else "MISS",
                       r["risk_expected"], r["risk"], "ok" if r["risk"] == r["risk_expected"] else "MISS",
                       ", ".join(r["flags"]) or "-"] for r in ev["rows"]])
    scores = table(["metric", "score"], [
        ["article accuracy", pct(s["article_correct"], n)],
        ["risk accuracy", pct(s["risk_correct"], n)],
        ["judge agreement, all cases", pct(s["judge_agree"], s["judge_cases"])],
        *[[f"  {check}: {kind}", pct(s["by_kind"][kind], n)] for check, kind in KINDS],
        ["real drafts the judge flagged", f'{s["real_drafts_flagged"]}/{n}'],
    ] + ([["llm calls / spend", f'{s["llm_calls"]} / ${s["llm_usd"]:.4f}']] if "llm_calls" in s else []))
    return f"Mode: {s['mode']}\n\n{per_note}\n\n{scores}"


def main():
    llm = llm_from_env()
    results = run(load_notes(), llm)
    ev = evaluate(results, llm)
    ev["summary"]["at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    out = HERE / "out"
    out.mkdir(exist_ok=True)
    (out / "run.json").write_text(json.dumps(to_run(results, llm), indent=1, ensure_ascii=False))
    (out / "eval.json").write_text(json.dumps(ev, indent=1, ensure_ascii=False))
    print(report(ev))


if __name__ == "__main__":
    main()
