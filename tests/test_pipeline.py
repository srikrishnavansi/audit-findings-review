"""One runnable check: drafting shape, judge flags, approve / edit / reject, the decision log,
the page guards, the eval score, and the LLM path against a local stub of the Messages API.

    python -m pytest -q
"""
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from fastapi.testclient import TestClient

import evals
import pipeline
import server

NOTES = pipeline.load_notes()
RESULTS = pipeline.run(NOTES)  # offline rules mode, no key needed
BASE, ORIGIN = "http://127.0.0.1:8765", {"Origin": "http://127.0.0.1:8765"}


def test_draft_shape():
    assert len(RESULTS) == len(NOTES) >= 6
    for r in RESULTS:
        d, note = r["draft"], r["note"]["text"]
        assert set(d) == {"title", "article", "risk", "evidence", "recommendation"}
        assert d["article"] in pipeline.ARTICLES and d["risk"] in pipeline.RISKS
        assert all(part.strip() in note for part in d["evidence"].split(";"))  # evidence is quoted from the note
        assert set(r["judge"]) == set(pipeline.CHECKS)


def test_judge_flags():
    note, d = RESULTS[0]["note"]["text"], RESULTS[0]["draft"]
    assert pipeline.flags(pipeline.rule_judge(note, d)) == []
    assert pipeline.flags(pipeline.rule_judge(note, {**d, "article": "Art. 35"})) == ["citation"]
    assert pipeline.flags(pipeline.rule_judge(note, {**d, "evidence": d["evidence"] + evals.INVENTED_FIGURE})) == ["faithful"]
    assert pipeline.flags(pipeline.rule_judge(note, {**d, "recommendation": evals.VAGUE})) == ["actionable"]
    assert pipeline.rule_judge(note, {**d, "article": "Art. 99"})["citation"]["why"] == "Art. 99 is not in the corpus"


@pytest.fixture
def app(tmp_path):
    run = pipeline.to_run(RESULTS)
    run["findings"][0]["note"] = "<script>alert(1)</script> & co"
    (tmp_path / "run.json").write_text(json.dumps(run))
    (tmp_path / "eval.json").write_text(json.dumps(evals.evaluate(RESULTS)))
    log = tmp_path / "decisions.jsonl"
    client = TestClient(server.create_app(tmp_path / "run.json", tmp_path / "eval.json", log), base_url=BASE)
    return client, log, run


def test_approve_edit_reject_and_append_only_log(app):
    client, log, run = app
    post = lambda body: client.post("/api/decisions", json=body, headers=ORIGIN)
    f0, f1, f2 = (f["draft"] for f in run["findings"][:3])

    assert post({"finding_id": "N01", "reviewer": "Lena Berger", "action": "approve"}).status_code == 200
    first_line = log.read_text()
    edited = {**f1, "risk": "medium", "recommendation": "DPO to complete a DPIA before go-live of the next release."}
    assert post({"finding_id": "N02", "reviewer": "Lena Berger", "action": "edit", "finding": edited}).status_code == 200
    assert post({"finding_id": "N03", "reviewer": "Lena Berger", "action": "reject", "note": "wrong scope"}).status_code == 200

    recs = [json.loads(x) for x in log.read_text().splitlines()]
    assert [(r["finding_id"], r["action"]) for r in recs] == [("N01", "approve"), ("N02", "edit"), ("N03", "reject")]
    assert recs[0]["final"] == f0 and recs[1]["final"] == edited and recs[1]["draft"] == f1 and recs[2]["final"] is None
    assert all(r["reviewer"] == "Lena Berger" and r["at"].endswith("Z") for r in recs)
    assert log.read_text().startswith(first_line)  # earlier lines untouched

    bad = [({"finding_id": "N04", "reviewer": "  ", "action": "approve"}, 422),           # no named reviewer
           ({"finding_id": "N04", "reviewer": "L", "action": "edit"}, 422),                # edit without a finding
           ({"finding_id": "N04", "reviewer": "L", "action": "edit", "finding": {**f2, "article": "Art. 99"}}, 422),
           ({"finding_id": "N04", "reviewer": "L", "action": "edit", "finding": run["findings"][3]["draft"]}, 422),
           ({"finding_id": "N04", "reviewer": "L", "action": "delete"}, 422),
           ({"finding_id": "N04", "reviewer": "L", "action": "approve", "extra": 1}, 422),
           ({"finding_id": "N99", "reviewer": "L", "action": "approve"}, 404)]
    for body, status in bad:
        assert post(body).status_code == status, body
    assert len(log.read_text().splitlines()) == 3  # refused decisions are not logged

    page = client.get("/").text
    assert "Approved by Lena Berger" in page and "Edited by Lena Berger" in page and "Rejected by Lena Berger" in page

    run["findings"][0]["draft"] = {**f0, "risk": "high"}  # a re-run changed the approved draft
    (log.parent / "run.json").write_text(json.dumps(run))
    assert "Approved by Lena Berger" not in client.get("/").text  # back in the queue, old approval stays in the log


def test_guards_and_escaping(app):
    client, log, _ = app
    body = {"finding_id": "N01", "reviewer": "L", "action": "approve"}
    assert client.post("/api/decisions", json=body).status_code == 403                                   # no Origin
    assert client.post("/api/decisions", json=body, headers={"Origin": "http://evil.test"}).status_code == 403
    assert client.post("/api/decisions", content=json.dumps(body),
                       headers={**ORIGIN, "content-type": "text/plain"}).status_code == 415
    assert client.get("/", headers={"Host": "evil.test"}).status_code == 403                             # DNS rebinding
    assert not log.exists()
    r = client.get("/")
    assert "<script>alert(1)</script>" not in r.text and "&lt;script&gt;alert(1)&lt;/script&gt; &amp; co" in r.text
    assert "script-src 'nonce-" in r.headers["content-security-policy"]
    assert "Judge agreement" in client.get("/eval").text


def test_eval_score():
    s = evals.evaluate(RESULTS)["summary"]
    assert (s["article_correct"], s["risk_correct"], s["notes"]) == (10, 10, 10)
    assert (s["judge_agree"], s["judge_cases"]) == (35, 50)
    assert s["by_kind"] == {"real draft": 10, "near-miss article": 5, "invented figure": 10,
                            "invented claim": 0, "vague recommendation": 10}
    table = evals.report(evals.evaluate(RESULTS))
    assert "article accuracy                    10/10  100%" in table and "35/50  70%" in table


def test_llm_mode_against_local_stub():
    """Real Anthropic SDK, local fake endpoint: checks the request shape, parsing, spend and the budget cap."""
    import anthropic
    seen = []

    class Stub(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["content-length"])))
            seen.append((self.path, body))
            props = body["output_config"]["format"]["schema"]["properties"]
            if "article" in props:
                out = {"title": "Storage limitation: CRM", "article": props["article"]["enum"][0], "risk": "medium",
                       "evidence": "31/40 > 5 yrs old", "recommendation": "System owner to purge before the follow-up audit."}
            else:
                out = {c: {"ok": True, "why": "fine"} for c in pipeline.CHECKS}
            data = json.dumps({"id": "msg_stub", "type": "message", "role": "assistant", "model": body["model"],
                               "content": [{"type": "text", "text": json.dumps(out)}], "stop_reason": "end_turn",
                               "stop_sequence": None, "usage": {"input_tokens": 1000, "output_tokens": 200}}).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    stub = HTTPServer(("127.0.0.1", 0), Stub)
    threading.Thread(target=stub.serve_forever, daemon=True).start()
    try:
        client = anthropic.Anthropic(api_key="stub-key", base_url=f"http://127.0.0.1:{stub.server_port}", max_retries=0)
        llm = pipeline.Claude(client)
        r = pipeline.run(NOTES[:1], llm)[0]
        assert r["draft"]["article"] == r["candidates"][0][0] and pipeline.flags(r["judge"]) == []
        assert llm.calls == 2 and llm.spent == pytest.approx(2 * (1000 * 0.10 + 200 * 0.50) / 1e6)
        for path, body in seen:
            assert path.endswith("/v1/messages") and body["model"] == "claude-haiku-5-5"
            assert "temperature" not in body and body["output_config"]["format"]["type"] == "json_schema"
        with pytest.raises(RuntimeError, match="budget"):
            pipeline.Claude(client, budget_usd=0).json("s", "p", pipeline.JUDGE_SCHEMA)
    finally:
        stub.shutdown()
