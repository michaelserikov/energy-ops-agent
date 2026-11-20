"""Agent loop: a planner chooses tool calls, tools return numbers, the answer is checked against them.

Two planners share one interface. RulePlanner is deterministic and needs no key. LLMPlanner wraps any
chat model with function calling. The
grounding check is the same for both: every number in the final text must occur in a tool result.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from . import tools as T

MAX_STEPS = 6
ALERT_HOURS = 8  # unusual hours in one day before operations is mailed (4% of days in 2024-2025)
DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


@dataclass
class Trace:
    calls: list = field(default_factory=list)  # (tool, args, result)
    answer: str = ""
    grounded: bool = True
    ungrounded_numbers: list = field(default_factory=list)


class RulePlanner:
    """Maps a question to tool calls by keywords and the dates in the text."""

    def plan(self, question: str) -> list[tuple[str, dict]]:
        dates = DATE.findall(question)
        q = question.lower()
        if dates and any(w in q for w in ("notice", "customer message")):
            return [("draft_notice", {"day": dates[0]})]
        kb_words = ("what is", "what are", "why", "how do", "when does", "runbook", "rule", "explain", "what do i do")
        if any(w in q for w in kb_words) and not any(w in q for w in ("summary", "report", "how was")):
            return [("search_kb", {"query": question})]
        if not dates:
            return []
        calls: list[tuple[str, dict]] = []
        if len(dates) >= 2 and any(w in q for w in ("compare", "versus", "vs", "difference")):
            return [("compare_days", {"day_a": dates[0], "day_b": dates[1]})]
        if len(dates) >= 2 and any(w in q for w in ("slope", "sensitiv", "per gw")):
            return [("wind_price_slope", {"start": dates[0], "end": dates[1]})]
        d = dates[0]
        if any(w in q for w in ("quality", "missing", "complete", "check")):
            calls.append(("data_quality", {"day": d}))
        if any(w in q for w in ("anomal", "unusual", "spike", "odd")):
            calls.append(("price_anomalies", {"day": d}))
        if not calls or any(w in q for w in ("summary", "report", "how was", "what happened")):
            calls.insert(0, ("daily_summary", {"day": d}))
        return calls


class LLMPlanner:  # pragma: no cover - needs a model endpoint
    """Tool use through any chat model that supports function calling.

    `chat(messages, tools)` must return a dict {"text": str, "tool_calls": [{"id", "name", "args"}]}.
    Adapters for a specific provider are thin wrappers around this callable. Not run in this repository.
    """

    def __init__(self, chat):
        self.chat = chat

    def tool_specs(self):
        return [{"name": n, "description": s["doc"],
                 "parameters": {"type": "object",
                                "properties": {k: {"type": "string", "description": v} for k, v in s["args"].items()},
                                "required": list(s["args"])}} for n, s in T.REGISTRY.items()]

    def run(self, question: str, trace: Trace, execute) -> str:
        d = T.load()
        span = f"{d.index.min().tz_convert('UTC'):%Y-%m-%d} to {d.index.max().tz_convert('UTC'):%Y-%m-%d}"
        msgs = [{"role": "system", "content": "You answer questions about German power data using the tools. "
                 "Quote numbers only as the tools return them. If the data cannot answer, say so. "
                 f"The data covers {span}. You do not know today's date and there is no forecast tool: "
                 "for questions about the future or about days outside the data, say that the tools cannot answer, "
                 "and never present a past day as a forecast."},
                {"role": "user", "content": question}]
        for _ in range(MAX_STEPS):
            r = self.chat(msgs, self.tool_specs())
            calls = r.get("tool_calls") or []
            if not calls:
                return r.get("text", "")
            msgs.append({"role": "assistant", "content": r.get("text", ""), "tool_calls": calls})
            for c in calls:
                out = execute(c["name"], dict(c["args"]))
                msgs.append({"role": "tool", "tool_call_id": c["id"], "content": json.dumps(out)})
        return "Stopped: too many tool calls."


def _numbers(text: str) -> list[str]:
    text = re.sub(r"(?m)^\s*(?:[-*]\s*)?\d{1,2}[.)]\s", " ", text)   # list numbering is not a figure
    text = re.sub(r"(?<=\d)[,  ](?=\d{3}(?!\d))", "", text)   # thousands separators: 1,453 and 59 496
    return re.findall(r"-?\d+(?:\.\d+)?", DATE.sub(" ", text))


def _flat(obj) -> list[str]:
    out: list[str] = []
    if isinstance(obj, dict):
        for v in obj.values():
            out += _flat(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out += _flat(v)
    elif isinstance(obj, (int, float)):
        out += [f"{obj:g}", str(obj), f"{obj:.1f}", f"{obj:.2f}", f"{obj:.0f}"]
    elif isinstance(obj, str):
        out += re.findall(r"-?\d+(?:\.\d+)?", obj) + re.findall(r"\d+", obj)
    return out


def check_grounded(answer: str, calls: list) -> list[str]:
    """Numbers in the answer that no tool result contains (as written or rounded)."""
    pool = set()
    for _, args, res in calls:
        pool |= set(_flat(res)) | set(_flat(args))
    pool_f = []
    for p in pool:
        try:
            pool_f.append(float(p))
        except ValueError:
            pass
    bad = []
    for n in _numbers(answer):
        if n in pool:
            continue
        try:
            v = float(n)
        except ValueError:
            continue
        if any(abs(v - p) <= 0.5 * 10 ** -(len(n.split(".")[1]) if "." in n else 0) for p in pool_f):
            continue
        bad.append(n)
    return bad


def render(calls: list) -> str:
    """Deterministic answer text built from the tool results."""
    lines = []
    for name, _, res in calls:
        if name == "daily_summary" and "error" not in res:
            lines.append(
                f"{res['day']}: mean price {res['price_mean']} EUR/MWh (min {res['price_min']}, max {res['price_max']}), "
                f"{res['negative_price_hours']} hours with negative prices, wind {res['wind_mean_mw']:.0f} MW on average, "
                f"solar peak {res['solar_peak_mw']:.0f} MW, residual load {res['residual_load_mean_mw']:.0f} MW on average.")
        elif name == "data_quality":
            f = res["flags"]
            lines.append(f"{res['day']}: {res['hours']} of {res['expected_hours']} hours present; "
                         + ("no data problems found." if not f else "flags: " + ", ".join(f) + "."))
        elif name == "price_anomalies":
            if res["n_anomalies"] == 0:
                lines.append(f"{res['day']}: no unusual hourly prices against the previous 28 days.")
            else:
                items = "; ".join(f"hour {a['hour']} price {a['price']} (typical {a['typical']})" for a in res["anomalies"])
                lines.append(f"{res['day']}: {res['n_anomalies']} unusual hours: {items}.")
        elif name == "search_kb":
            if not res["hits"]:
                lines.append("The knowledge base has no passage on this; ask the process team.")
            for h in res["hits"]:
                lines.append(f"[{h['source']}] {h['text']}")
        elif name == "draft_notice":
            lines.append(("DRAFT, needs approval: " + res["text"]) if "text" in res else f"{name}: {res}")
        elif name == "compare_days":
            lines.append(f"Mean price {res['a']['day']} against {res['b']['day']}: {res['price_mean_diff']} EUR/MWh; "
                         f"mean wind difference {res['wind_mean_diff_mw']:.0f} MW.")
        elif name == "wind_price_slope" and "error" not in res:
            lines.append(f"{res['start']} to {res['end']}: price changes by {res['eur_per_mwh_per_gw']} EUR/MWh per GW of "
                         f"residual load (correlation {res['correlation']}, {res['hours']} hours).")
        else:
            lines.append(f"{name}: {res}")
    return "\n".join(lines)


class Agent:
    def __init__(self, planner=None, df=None):
        self.planner = planner or RulePlanner()
        self.df = df

    def execute(self, name: str, args: dict):
        if name not in T.REGISTRY:
            return {"error": f"unknown tool {name}"}
        spec = T.REGISTRY[name]
        if set(args) != set(spec["args"]):
            return {"error": f"bad arguments for {name}"}
        for k, v in args.items():
            if not isinstance(v, str):
                return {"error": f"{k} must be text"}
            if spec["args"][k] == "YYYY-MM-DD" and not DATE.fullmatch(v):
                return {"error": "dates must be YYYY-MM-DD"}
            if spec["args"][k] == "text" and len(v) > 300:
                return {"error": "query too long"}
        try:
            return spec["fn"](**args, df=self.df)
        except Exception as e:  # tools never crash the loop
            return {"error": f"{type(e).__name__}: {e}"}

    def ask(self, question: str) -> Trace:
        tr = Trace()

        def run(name, args):
            res = self.execute(name, args)
            tr.calls.append((name, args, res))
            return res

        if isinstance(self.planner, RulePlanner):
            plan = self.planner.plan(question)
            if not plan:
                tr.answer = "I need a date (YYYY-MM-DD) to look at the data."
                return tr
            for name, args in plan[:MAX_STEPS]:
                run(name, args)
            tr.answer = render(tr.calls)
        else:
            tr.answer = self.planner.run(question, tr, run)
        bad = check_grounded(tr.answer, tr.calls)
        tr.grounded, tr.ungrounded_numbers = (not bad), bad
        if bad:
            tr.answer = "Withheld: the draft answer contained numbers no tool returned (" + ", ".join(bad) + ")."
        return tr


def daily_report(day: str, df=None) -> dict:
    """Used by the scheduled workflow: summary, quality and anomaly flags in one call."""
    a = Agent(df=df)
    q = a.ask(f"Report for {day}: summary, data quality check and unusual price spikes")
    n_flags = 0
    n_quality = 0
    n_anom = 0
    has_data = all("error" not in res for name, _, res in q.calls if name == "daily_summary")
    for name, _, res in q.calls:
        if name == "data_quality":
            n_quality = len(res["flags"])
            n_flags += n_quality
        if name == "price_anomalies":
            n_anom = res["n_anomalies"]
            n_flags += n_anom
    return {"day": day, "has_data": has_data, "n_flags": n_flags, "alert": bool(n_quality > 0 or n_anom >= ALERT_HOURS), "text": q.answer, "grounded": q.grounded,
            "tools": [c[0] for c in q.calls]}
