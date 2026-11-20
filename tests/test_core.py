import numpy as np
import pandas as pd
import pytest

from eoa import tools as T
from eoa.agent import Agent, check_grounded, daily_report
from eoa.api import app


@pytest.fixture(scope="module")
def df():
    return T.load()


def test_known_event_dunkelflaute(df):
    s = T.daily_summary("2024-12-12", df)
    assert s["wind_mean_mw"] < 3000 and s["price_max"] > 900
    assert T.price_anomalies("2024-12-12", df)["n_anomalies"] >= 8


def test_injected_faults_found(df):
    d = df.copy()
    day = "2024-06-05"
    idx = d.loc[day].index
    d = d.drop(idx[[3, 4]])                         # two missing hours
    d.loc[idx[1], "solar"] = 9000.0                 # solar at night
    d.loc[idx[10], "load"] = np.nan                 # missing value
    f = T.data_quality(day, d)["flags"]
    assert any(x.startswith("missing_hours") for x in f)
    assert "solar_at_night" in f
    assert any(x.startswith("missing_values") for x in f)


def test_clean_day_has_no_flags(df):
    assert T.data_quality("2024-06-01", df)["flags"] == []


def test_grounding_blocks_invented_number(df):
    t = Agent(df=df).ask("How was 2024-06-01? summary")
    assert t.grounded
    assert check_grounded(t.answer, t.calls) == []
    assert check_grounded("The mean price was 42.17 EUR/MWh.", t.calls) == ["42.17"]


def test_unknown_tool_and_bad_args(df):
    a = Agent(df=df)
    assert "error" in a.execute("delete_everything", {})
    assert "error" in a.execute("daily_summary", {"day": "yesterday"})
    assert "error" in a.execute("daily_summary", {"day": "2024-06-01", "x": "1"})


def test_no_date_no_guess(df):
    t = Agent(df=df).ask("price please")
    assert t.calls == [] and "date" in t.answer


def test_kb_retrieval_finds_right_document(df):
    a = Agent(df=df)
    assert a.ask("When does operations get an alert?").calls[0][2]["hits"][0]["source"] == "04_alert_rules"
    assert a.ask("What is residual load?").calls[0][2]["hits"][0]["source"] == "01_residual_load"


def test_kb_out_of_scope_is_refused(df):
    t = Agent(df=df).ask("what is the capital of France")
    assert t.calls[0][2]["found"] is False and "no passage" in t.answer


def test_notice_needs_approval_and_is_grounded(df):
    t = Agent(df=df).ask("Draft a customer notice for 2024-12-12")
    assert t.calls[0][2]["needs_approval"] is True and t.grounded and t.answer.startswith("DRAFT")


def test_long_query_rejected(df):
    assert "error" in Agent(df=df).execute("search_kb", {"query": "x" * 500})


def test_report_and_api():
    from fastapi.testclient import TestClient
    c = TestClient(app)
    r = c.get("/report", params={"day": "2024-12-12"}).json()
    assert r["alert"] is True and r["grounded"]
    assert c.get("/report", params={"day": "2024-06-01"}).json()["alert"] is False
    assert c.get("/report", params={"day": "bad"}).status_code == 422
    assert c.get("/report", params={"day": "2030-01-01"}).status_code == 404
    assert c.post("/ask", json={"question": "check data quality 2024-06-01"}).json()["grounded"]


def test_llm_planner_with_scripted_model(df):
    """LLMPlanner loop with a scripted chat function: one tool call, then an answer."""
    from eoa.agent import LLMPlanner
    seen = []

    def chat(msgs, tools):
        seen.append(len(msgs))
        if not any(m["role"] == "tool" for m in msgs):
            return {"text": "", "tool_calls": [{"id": "c0", "name": "daily_summary", "args": {"day": "2024-06-01"}}]}
        return {"text": "Mean price on 2024-06-01 was 12345.67 EUR/MWh.", "tool_calls": []}   # invented number

    t = Agent(planner=LLMPlanner(chat), df=df).ask("How was 2024-06-01?")
    assert [c[0] for c in t.calls] == ["daily_summary"]
    assert not t.grounded and "12345.67" in t.answer and t.answer.startswith("Withheld")


def test_gemini_translation_roundtrip():
    from eoa.providers import from_gemini, to_gemini
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "q"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c0", "name": "daily_summary", "args": {"day": "2024-06-01"}}]},
            {"role": "tool", "tool_call_id": "c0", "content": "{\"price_mean\": 1.5}"}]
    tools = [{"name": "daily_summary", "description": "d", "parameters": {"type": "object", "properties": {}}}]
    b = to_gemini(msgs, tools)
    assert b["systemInstruction"]["parts"][0]["text"] == "s"
    assert b["contents"][1]["parts"][0]["functionCall"]["name"] == "daily_summary"
    assert b["contents"][2]["parts"][0]["functionResponse"]["name"] == "daily_summary"
    r = from_gemini({"candidates": [{"content": {"parts": [{"functionCall": {"name": "x", "args": {"a": "1"}}}]}}]})
    assert r["tool_calls"][0]["name"] == "x" and r["text"] == ""


def test_report_of_a_quiet_day_is_not_withheld(df):
    r = daily_report("2024-06-01", df)
    assert r["grounded"] and "Withheld" not in r["text"]


def test_every_day_of_a_month_is_grounded(df):
    for d in pd.date_range("2024-03-01", "2024-03-31"):
        assert daily_report(d.strftime("%Y-%m-%d"), df)["grounded"], d


def test_thousands_separators_do_not_split_numbers():
    calls = [("x", {}, {"wind_mean_mw": 1453.0, "residual": 59496.0})]
    assert check_grounded("Wind was 1,453 MW and residual load 59,496 MW.", calls) == []
    assert check_grounded("Wind was 1,454 MW.", calls) == ["1454"]


def test_list_numbering_is_not_a_figure():
    calls = [("x", {}, {"v": 8})]
    text = "\n".join(["Steps:", "1. Check the source.", "2. Repeat later.", "3. Tell the owner. Value 8."])
    assert check_grounded(text, calls) == []
    text = "\n".join(["Steps:", "1. Check the source. Value 9."])
    assert check_grounded(text, calls) == ["9"]
