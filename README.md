# Energy operations agent: checked answers and a daily report on German power data

A small agent that answers questions about German hourly power data (day-ahead price, wind, solar, load; SMARD, 2023-2025) by calling tools, and a daily report that a workflow engine can call every morning. The point is the control around the language model: tools with fixed arguments, a step limit, an audit trail of every call, and a check that every number in an answer occurs in a tool result. An answer with a number no tool returned is withheld.

## What it does

| Part | File | Notes |
|---|---|---|
| Tools | `src/eoa/tools.py` | data quality (missing or duplicate hours, missing values, solar at night, implausible values), daily summary, unusual hourly prices against the same hour in the previous 28 days (robust z-score), day comparison, price slope per GW of residual load |
| Agent loop | `src/eoa/agent.py` | `RulePlanner` (deterministic, no key) and `LLMPlanner` (any chat model with function calling, passed in as a callable). Arguments are validated (dates only), unknown tools are refused, tool errors do not stop the loop |
| Grounding check | `agent.check_grounded` | numbers in the answer must match a tool result or its rounding; otherwise the answer is replaced by a refusal that lists the numbers |
| HTTP | `src/eoa/api.py` | `GET /report?day=`, `POST /ask`, `GET /health` (FastAPI) |
| Workflow | `n8n/daily_energy_report.json` | 07:00 schedule, yesterday's date, call `/report` with three retries, branch on `alert`, mail operations or log |

## Results

- **Known event recovered.** 12 December 2024 (dark doldrums): mean price 395 EUR/MWh, maximum 936, wind 1.5 GW on average; 14 hours flagged as unusual.
- **Injected faults found.** Tests drop hours, set a missing value and put solar output at night on a real day; all three are flagged. A clean day has no flags.
- **A naive alarm is too noisy.** At least one unusual hour occurs on 32% of days (223 of 702 days, 2024-2025), so mailing on any flag would mail every third day. The workflow alerts on a data fault or at least 8 unusual hours in a day: 29 of 702 days (4%). No data faults occur in this data set (the SMARD series is clean), so the quality checks are tested only on injected faults. See `results/flag_backtest.json`, `scripts/backtest_flags.py`.
- **A bug the tests caught.** The first version decided "no data for this day" by searching the report text for the words "no data", which the sentence "no data problems found" also contains; every day returned 404. The check now uses the tool result.
- **Price slope.** January to March 2025: 3.1 EUR/MWh per GW of residual load (correlation 0.88 over 2,136 hours), a descriptive number.

## Not done

- `LLMPlanner` is written against a generic function-calling interface but has not been run with a real model: no model endpoint was available. Only the deterministic planner is tested.
- `n8n/daily_energy_report.json` is valid JSON in the n8n node format but has not been imported or executed: n8n is not installed on the development machine. Mail addresses are placeholders.
- Data is a stored SMARD export, not a live feed; tools never write or send anything, and no personal or customer data is involved.
- Research and demonstration project, not an operational monitoring system.

## Run

```bash
pip install -e ".[test]"
pytest                                  # 7 tests
python scripts/backtest_flags.py
uvicorn eoa.api:app --port 8000         # then GET /report?day=2024-12-12
```

## Data licence

SMARD.de, Bundesnetzagentur (CC BY 4.0): hourly day-ahead price, generation and load for Germany, 2023-2025, stored in `data/`.
