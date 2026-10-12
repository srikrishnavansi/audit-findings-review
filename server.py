"""Reviewer screen: raw note and draft finding side by side, with the judge's flags.
A named reviewer approves, edits or rejects; every decision is appended to out/decisions.jsonl.
Local only: binds 127.0.0.1, checks Host and Origin, accepts JSON-only POSTs, escapes every
rendered value and sends a nonce CSP.

    python server.py [port]      # then open http://127.0.0.1:8765
"""
import html
import json
import secrets
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from pipeline import ARTICLES, CHECKS, HERE, RISKS

MAX_BODY = 64 * 1024


class Finding(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str = Field(min_length=1, max_length=300)
    article: str = Field(max_length=40)
    risk: Literal["low", "medium", "high"]
    evidence: str = Field(min_length=1, max_length=4000)
    recommendation: str = Field(min_length=1, max_length=2000)


class Decision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    finding_id: str = Field(max_length=20)
    reviewer: str = Field(min_length=1, max_length=80)
    action: Literal["approve", "edit", "reject"]
    finding: Finding | None = None
    note: str = Field(default="", max_length=1000)


def e(v):
    return html.escape(str(v), quote=True)


def read_json(path):
    return json.loads(Path(path).read_text()) if Path(path).exists() else None


def decisions(log_path):
    if not Path(log_path).exists():
        return []
    return [json.loads(line) for line in Path(log_path).read_text().splitlines() if line.strip()]


# ---------- rendering ----------

CSS = """
:root{--bg:#f5f5f2;--card:#fff;--ink:#1b1b1a;--mut:#66665f;--line:#e3e2dc;--soft:#f1f0eb;--amber:#9a5200;--amber-bg:#fff3dc;
--green:#1b7543;--green-bg:#e5f3ea;--red:#b42318;--red-bg:#fdecea;--blue:#1f5fbf;--blue-bg:#e8effa}
@media (prefers-color-scheme:dark){:root{--bg:#141413;--card:#1c1c1a;--ink:#ecebe6;--mut:#a3a29b;--line:#33322e;--soft:#252522;
--amber:#f0a14a;--amber-bg:#3a2a12;--green:#5cc98b;--green-bg:#15301f;--red:#f2867c;--red-bg:#3a1a17;--blue:#86adf0;--blue-bg:#18263d}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14.5px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,sans-serif}
main{max-width:1320px;margin:0 auto;padding:26px 16px 48px}
header{display:flex;flex-wrap:wrap;align-items:flex-end;justify-content:space-between;gap:14px;margin-bottom:18px}
h1{font-size:24px;margin:0 0 4px;letter-spacing:-.01em}.sub{margin:0;color:var(--mut);max-width:680px}
.bar{display:flex;flex-wrap:wrap;gap:8px;align-items:center}
.stat{background:var(--card);border:1px solid var(--line);border-radius:999px;padding:4px 12px;font-size:13px;color:var(--ink);text-decoration:none}
.stat b{font-variant-numeric:tabular-nums}.s-pending b{color:var(--amber)}.s-approve b,.s-edit b{color:var(--green)}.s-reject b{color:var(--red)}
.who{display:flex;flex-wrap:wrap;align-items:center;gap:8px;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:8px 12px;margin-bottom:16px}
.who label{font-weight:600;font-size:13px;white-space:nowrap}.who input{flex:1 1 200px;max-width:320px;margin:0}.who span{color:var(--mut);font-size:12.5px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;margin-bottom:14px;overflow:hidden}
.card.flagged{border-left:4px solid var(--amber)}
.top{display:flex;flex-wrap:wrap;align-items:center;gap:8px;padding:10px 16px;border-bottom:1px solid var(--line);background:var(--soft)}
.id{font-weight:700;font-variant-numeric:tabular-nums}.top .t{color:var(--mut);font-size:13px;flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.pill{display:inline-block;border-radius:6px;padding:1px 8px;font-size:12px;font-weight:600}
.p-pending{color:var(--amber);background:var(--amber-bg)}.p-approve,.p-edit{color:var(--green);background:var(--green-bg)}.p-reject{color:var(--red);background:var(--red-bg)}
.p-syn{color:var(--blue);background:var(--blue-bg);font-weight:500}
.cols{display:grid;grid-template-columns:minmax(0,.9fr) minmax(0,1.35fr) minmax(0,1fr)}
.cols>section{padding:14px 16px;min-width:0}.cols>section+section{border-left:1px solid var(--line)}
@media (max-width:1060px){.cols{grid-template-columns:1fr}.cols>section+section{border-left:0;border-top:1px solid var(--line)}}
h3{font-size:11.5px;text-transform:uppercase;letter-spacing:.06em;color:var(--mut);margin:0 0 8px;font-weight:600}
pre.note{margin:0;white-space:pre-wrap;overflow-wrap:anywhere;font:12.5px/1.55 ui-monospace,SFMono-Regular,Menlo,monospace;background:var(--soft);border-radius:8px;padding:10px 12px}
label.f{display:block;font-size:12px;color:var(--mut);margin-bottom:8px}
input,select,textarea{display:block;width:100%;min-width:0;margin-top:3px;font:inherit;font-size:13.5px;color:var(--ink);background:var(--card);border:1px solid var(--line);border-radius:7px;padding:6px 9px}
textarea{resize:vertical}.row{display:grid;grid-template-columns:minmax(0,1fr) 110px;gap:10px}
input:focus-visible,select:focus-visible,textarea:focus-visible,button:focus-visible,a:focus-visible{outline:2px solid var(--blue);outline-offset:1px}
blockquote{margin:2px 0 0;padding:8px 10px;border-left:3px solid var(--line);color:var(--mut);font-size:12.5px;background:var(--soft);border-radius:0 6px 6px 0}
ul.checks{list-style:none;margin:0 0 12px;padding:0}ul.checks li{display:grid;grid-template-columns:62px minmax(0,1fr);gap:2px 8px;padding:7px 0;border-top:1px solid var(--line);font-size:13px}
ul.checks li:first-child{border-top:0}.v{font-size:11.5px;font-weight:700;border-radius:5px;padding:1px 0;text-align:center;height:fit-content}
.v.ok{color:var(--green);background:var(--green-bg)}.v.flag{color:var(--amber);background:var(--amber-bg)}.why{color:var(--mut)}.ck{font-weight:600;color:var(--ink)}
.cands{font-size:12px;color:var(--mut);margin:0 0 12px}.cands code{font-size:11.5px}
.actions{display:flex;flex-wrap:wrap;gap:8px;align-items:center}.actions input{flex:1 1 100%;margin:0}
button{font:inherit;font-size:13.5px;border-radius:8px;padding:7px 13px;border:1px solid var(--line);background:var(--card);color:var(--ink);cursor:pointer}
button.ok{background:var(--green);border-color:var(--green);color:#fff}button.no{color:var(--red);border-color:var(--red)}
button:disabled{opacity:.4;cursor:default}.msg{flex-basis:100%;font-size:12.5px;color:var(--red)}.msg:empty{display:none}
.decided{font-size:12.5px;color:var(--mut);margin:0 0 10px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin-bottom:18px}
.tile{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px}.tile .k{font-size:12px;color:var(--mut)}
.tile .n{font-size:28px;font-weight:650;font-variant-numeric:tabular-nums;letter-spacing:-.02em}.tile .d{font-size:12.5px;color:var(--mut)}
h2{font-size:16px;margin:22px 0 8px}.wrap{overflow-x:auto}
table{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--line);border-radius:12px;font-size:13px}
th,td{text-align:left;padding:7px 10px;border-top:1px solid var(--line);font-variant-numeric:tabular-nums}
th{border-top:0;color:var(--mut);font-weight:500;font-size:12px}td.miss{color:var(--red);font-weight:600}td.hit{color:var(--green)}
progress{appearance:none;width:120px;height:8px;border:0;border-radius:4px;background:var(--soft);vertical-align:middle;margin-right:8px;overflow:hidden}
progress::-webkit-progress-bar{background:var(--soft)}progress::-webkit-progress-value{background:var(--green)}progress::-moz-progress-bar{background:var(--green)}
progress.lo::-webkit-progress-value{background:var(--amber)}progress.lo::-moz-progress-bar{background:var(--amber)}
.foot{margin-top:14px;font-size:12.5px;color:var(--mut)}a{color:var(--blue)}
"""

JS = """
const who = document.getElementById('reviewer');
try { who.value = localStorage.getItem('reviewer') || ''; } catch (_) {}
who.addEventListener('input', () => { try { localStorage.setItem('reviewer', who.value); } catch (_) {} });
const FIELDS = ['title', 'article', 'risk', 'evidence', 'recommendation'];
const read = card => Object.fromEntries(FIELDS.map(f => [f, card.querySelector('[name=' + f + ']').value]));
const sync = card => {
  const edited = JSON.stringify(read(card)) !== card.dataset.original;
  card.querySelector('[data-action=edit]').disabled = !edited;
  card.querySelector('[data-action=approve]').disabled = edited;
};
document.querySelectorAll('[data-finding]').forEach(card => {
  card.dataset.original = JSON.stringify(read(card));
  card.addEventListener('input', () => sync(card));
});
document.addEventListener('click', async ev => {
  const b = ev.target.closest('button[data-action]');
  if (!b) return;
  const card = b.closest('[data-finding]'), msg = card.querySelector('.msg');
  if (!who.value.trim()) { msg.textContent = 'Enter your name at the top first.'; who.focus(); return; }
  const body = {finding_id: card.dataset.finding, reviewer: who.value.trim(), action: b.dataset.action,
                note: card.querySelector('.note').value};
  if (body.action === 'edit') body.finding = read(card);
  card.querySelectorAll('button').forEach(x => { x.disabled = true; });
  const r = await fetch('/api/decisions', {method: 'POST', headers: {'content-type': 'application/json'},
                                          body: JSON.stringify(body)});
  if (r.ok) return location.reload();
  const j = await r.json().catch(() => ({}));
  msg.textContent = j.error || (Array.isArray(j.detail) ? j.detail.map(d => d.msg).join('; ') : 'Request failed');
  card.querySelector('[data-action=reject]').disabled = false;
  sync(card);
});
"""

NAMES = {"citation": "Citation", "actionable": "Actionable", "faithful": "No invented facts"}
DONE = {"approve": "Approved", "edit": "Edited", "reject": "Rejected"}


def page(title, nonce, body, script=""):
    js = f'<script nonce="{nonce}">{script}</script>' if script else ""
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1"><title>{e(title)}</title>'
            f'<style nonce="{nonce}">{CSS}</style></head><body><main>{body}</main>{js}</body></html>')


def options(values, selected, label=lambda v: v):
    return "".join(f'<option value="{e(v)}"{" selected" if v == selected else ""}>{e(label(v))}</option>' for v in values)


def card(f, last):
    shown = last["final"] if last and last["action"] == "edit" else f["draft"]
    art = ARTICLES.get(shown["article"], {"title": "not in the corpus", "text": ""})
    status = (f'<span class="pill p-{e(last["action"])}">{DONE[last["action"]]} by {e(last["reviewer"])}</span>'
              f'<span class="t">{e(last["at"][11:16])} UTC{" &middot; " + e(last["note"]) if last["note"] else ""}</span>'
              if last else '<span class="pill p-pending">Waiting for review</span><span class="t"></span>')
    checks = "".join(
        f'<li><span class="v {"ok" if f["judge"][c]["ok"] else "flag"}">{"PASS" if f["judge"][c]["ok"] else "FLAG"}</span>'
        f'<span><span class="ck">{NAMES[c]}</span> <span class="why">{e(f["judge"][c]["why"])}</span></span></li>'
        for c in CHECKS)
    cands = " &middot; ".join(f"<code>{e(a)}</code> {e(s)}" for a, s in f["candidates"])
    fid = e(f["id"])
    return f"""
<article class="card{" flagged" if f["flags"] else ""}" data-finding="{fid}">
  <div class="top"><span class="id">{fid}</span>{status}</div>
  <div class="cols">
    <section><h3>Raw note <span class="pill p-syn">synthetic</span></h3><pre class="note">{e(f["note"])}</pre></section>
    <section><h3>Draft finding{" &middot; showing the reviewer's edit" if last and last["action"] == "edit" else ""}</h3>
      <label class="f">Title<input name="title" value="{e(shown["title"])}" maxlength="300"></label>
      <div class="row">
        <label class="f">GDPR article<select name="article">{options(ARTICLES, shown["article"], lambda a: f'{a} {ARTICLES[a]["title"]}')}</select></label>
        <label class="f">Risk<select name="risk">{options(RISKS, shown["risk"])}</select></label>
      </div>
      <label class="f">Evidence<textarea name="evidence" rows="3" maxlength="4000">{e(shown["evidence"])}</textarea></label>
      <label class="f">Recommendation<textarea name="recommendation" rows="3" maxlength="2000">{e(shown["recommendation"])}</textarea></label>
      <label class="f">Cited text, {e(shown["article"])} {e(art["title"])}<blockquote>{e(art["text"])}</blockquote></label>
    </section>
    <section><h3>Judge, on the draft</h3><ul class="checks">{checks}</ul>
      <p class="cands">Retrieved: {cands}</p>
      <div class="actions">
        <input class="note" aria-label="Note for the decision log" placeholder="Note for the decision log (optional)" maxlength="1000">
        <button class="ok" data-action="approve">Approve</button>
        <button data-action="edit" disabled>Save edits</button>
        <button class="no" data-action="reject">Reject</button>
        <span class="msg" role="status"></span>
      </div>
    </section>
  </div>
</article>"""


def review_page(run, log, nonce):
    if not run:
        return page("Audit findings review", nonce, '<h1>No run yet</h1><p class="sub">Run <code>python evals.py</code> first.</p>')
    last = {d["finding_id"]: d for d in log}  # later lines win; the log itself keeps every decision
    # a decision covers the exact draft it was made on: a re-drafted finding goes back to the queue
    latest = {f["id"]: last[f["id"]] for f in run["findings"] if f["id"] in last and last[f["id"]]["draft"] == f["draft"]}
    counts = {a: sum(d["action"] == a for d in latest.values()) for a in DONE}
    pending = len(run["findings"]) - sum(counts.values())
    stats = (f'<span class="stat s-pending"><b>{pending}</b> waiting</span>'
             + "".join(f'<span class="stat s-{a}"><b>{counts[a]}</b> {DONE[a].lower()}</span>' for a in DONE)
             + '<a class="stat" href="/eval">Eval report</a>')
    cards = "".join(card(f, latest.get(f["id"])) for f in run["findings"])
    body = f"""
<header><div><h1>Audit findings review</h1>
<p class="sub">Synthetic GDPR audit notes, drafted and checked by {e(run["mode"])}. A finding counts as final only
after a named reviewer approves or edits it. Every decision is appended to <code>out/decisions.jsonl</code>.</p></div>
<div class="bar">{stats}</div></header>
<div class="who"><label for="reviewer">Reviewing as</label><input id="reviewer" maxlength="80" placeholder="Your full name" autocomplete="name">
<span>Logged with every decision.</span></div>
{cards}"""
    return page("Audit findings review", nonce, body, JS)


def bar(a, b):
    return f'<progress class="{"lo" if a < 0.8 * b else ""}" value="{a}" max="{b}"></progress>{a}/{b}'


def eval_page(ev, nonce):
    if not ev:
        return page("Eval report", nonce, '<h1>No eval yet</h1><p class="sub">Run <code>python evals.py</code> first.</p>')
    s, n = ev["summary"], ev["summary"]["notes"]
    tiles = "".join(f'<div class="tile"><div class="k">{k}</div><div class="n">{a}/{b}</div><div class="d">{d}</div></div>' for k, a, b, d in (
        ("Article accuracy", s["article_correct"], n, "drafted article = hand label"),
        ("Risk accuracy", s["risk_correct"], n, "drafted risk = hand label"),
        ("Judge agreement", s["judge_agree"], s["judge_cases"], "judge verdict = known truth"),
        ("Drafts the judge flagged", s["real_drafts_flagged"], n, "sent to the reviewer with a flag")))
    explain = {"real draft": "Is the cited article right? Truth from the hand label.",
               "near-miss article": "Draft re-cited to the runner-up retrieved article. Should be flagged.",
               "invented figure": "A made-up vendor name, count and date appended to the evidence.",
               "invented claim": "A made-up claim in plain words appended to the evidence.",
               "vague recommendation": "Recommendation replaced by a hedge with no owner or deadline."}
    kinds = {c["kind"]: c["check"] for c in ev["cases"]}
    by_kind = "".join(f'<tr><td>{e(k)}</td><td>{e(kinds[k])}</td><td>{bar(s["by_kind"][k], n)}</td><td>{e(explain[k])}</td></tr>'
                      for k in s["by_kind"])
    rows = "".join(
        f'<tr><td>{e(r["id"])}</td><td>{e(r["article_expected"])}</td>'
        f'<td class="{"hit" if r["article"] == r["article_expected"] else "miss"}">{e(r["article"])}</td>'
        f'<td>{e(r["risk_expected"])}</td><td class="{"hit" if r["risk"] == r["risk_expected"] else "miss"}">{e(r["risk"])}</td>'
        f'<td>{e(", ".join(r["flags"]) or "none")}</td></tr>' for r in ev["rows"])
    spend = f' &middot; {s["llm_calls"]} LLM calls, ${s["llm_usd"]:.4f}' if "llm_calls" in s else ""
    body = f"""
<header><div><h1>Eval report</h1><p class="sub">{n} synthetic notes with hand labels. Mode: {e(s["mode"])}{spend}
&middot; {e(s.get("at", ""))}</p></div><div class="bar"><a class="stat" href="/">Back to review</a></div></header>
<div class="tiles">{tiles}</div>
<h2>Judge agreement by case type</h2>
<div class="wrap"><table><thead><tr><th>Case</th><th>Check</th><th>Agreement</th><th>What it tests</th></tr></thead><tbody>{by_kind}</tbody></table></div>
<h2>Per note</h2>
<div class="wrap"><table><thead><tr><th>Note</th><th>Expected article</th><th>Drafted</th><th>Expected risk</th><th>Drafted</th><th>Judge flags</th></tr></thead>
<tbody>{rows}</tbody></table></div>
<p class="foot">The offline rules and the hand labels come from the same author, so the drafting scores are a floor check.
The judge rows show where lexical checks stop: near-miss citations and invented claims in plain words are what the LLM judge is for.</p>"""
    return page("Eval report", nonce, body)


# ---------- app ----------

def create_app(run_path=HERE / "out/run.json", eval_path=HERE / "out/eval.json",
               log_path=HERE / "out/decisions.jsonl", port=8765):
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
    origins = {"http://" + h for h in hosts}
    lock = threading.Lock()

    @app.middleware("http")
    async def guard(request: Request, call_next):
        if request.headers.get("host") not in hosts:
            return JSONResponse({"error": "bad Host header"}, 403)  # blocks DNS rebinding
        if request.method == "POST":
            if request.headers.get("origin") not in origins:
                return JSONResponse({"error": "bad or missing Origin header"}, 403)  # CSRF
            if request.headers.get("content-type", "").split(";")[0].strip().lower() != "application/json":
                return JSONResponse({"error": "send application/json"}, 415)
            if not request.headers.get("content-length", "").isdigit():
                return JSONResponse({"error": "content-length required"}, 411)
            if int(request.headers["content-length"]) > MAX_BODY:
                return JSONResponse({"error": "body too large"}, 413)
        request.state.nonce = secrets.token_urlsafe(16)
        resp = await call_next(request)
        n = request.state.nonce
        resp.headers.update({
            "content-security-policy": "default-src 'none'; connect-src 'self'; base-uri 'none'; form-action 'none'; "
                                       f"frame-ancestors 'none'; script-src 'nonce-{n}'; style-src 'nonce-{n}'",
            "x-content-type-options": "nosniff", "referrer-policy": "no-referrer", "cache-control": "no-store"})
        return resp

    @app.get("/", response_class=HTMLResponse)
    def review(request: Request):
        return review_page(read_json(run_path), decisions(log_path), request.state.nonce)

    @app.get("/eval", response_class=HTMLResponse)
    def evals(request: Request):
        return eval_page(read_json(eval_path), request.state.nonce)

    @app.post("/api/decisions")
    def decide(d: Decision):
        run = read_json(run_path) or {"findings": [], "mode": ""}
        f = next((x for x in run["findings"] if x["id"] == d.finding_id), None)
        reviewer = d.reviewer.strip()
        if not f:
            return JSONResponse({"error": "unknown finding"}, 404)
        if not reviewer:
            return JSONResponse({"error": "a named reviewer is required"}, 422)
        final = f["draft"] if d.action == "approve" else None
        if d.action == "edit":
            final = d.finding and d.finding.model_dump()
            if not final:
                return JSONResponse({"error": "an edit needs the edited finding"}, 422)
            if final["article"] not in ARTICLES:
                return JSONResponse({"error": "article is not in the corpus"}, 422)
            if final == f["draft"]:
                return JSONResponse({"error": "nothing changed, use approve"}, 422)
        rec = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
               "finding_id": f["id"], "reviewer": reviewer, "action": d.action, "note": d.note.strip(),
               "mode": run["mode"], "judge_flags": f["flags"], "draft": f["draft"], "final": final}
        with lock, Path(log_path).open("a", encoding="utf-8") as fh:  # append-only
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    return app


if __name__ == "__main__":
    import uvicorn
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    print(f"Reviewer screen: http://127.0.0.1:{port}/  (eval report at /eval)", flush=True)
    uvicorn.run(create_app(port=port), host="127.0.0.1", port=port, log_level="warning")  # loopback only
