"""HTTP interface for the scheduled workflow (n8n calls /report every morning)."""
from fastapi import FastAPI, HTTPException, Query

from .agent import Agent, daily_report

app = FastAPI(title="energy-ops-agent")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/report")
def report(day: str = Query(pattern=r"^\d{4}-\d{2}-\d{2}$")):
    r = daily_report(day)
    if not r["has_data"]:
        raise HTTPException(status_code=404, detail="no data for this day")
    return r


@app.post("/ask")
def ask(body: dict):
    q = str(body.get("question", ""))[:500]
    t = Agent().ask(q)
    return {"answer": t.answer, "grounded": t.grounded, "calls": [{"tool": c[0], "args": c[1]} for c in t.calls]}


@app.post("/outbox")
def outbox(body: dict):
    """Stand-in for a mail server: appends the message to a local file. Nothing leaves the machine."""
    import json, time
    from pathlib import Path
    p = Path(__file__).resolve().parents[2] / "results" / "outbox.jsonl"
    rec = {"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "subject": str(body.get("subject", ""))[:200], "text": str(body.get("text", ""))[:4000]}
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    return {"stored": True}
