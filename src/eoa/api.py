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
