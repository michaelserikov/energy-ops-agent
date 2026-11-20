"""Tools the agent may call. Every tool returns a plain dict of numbers and flags.

Data: hourly SMARD series for Germany (day-ahead price, wind, solar, load), 2023-2025.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[2] / "data" / "smard_hourly_2023_2025.parquet"


@lru_cache(maxsize=1)
def load(path: str | None = None) -> pd.DataFrame:
    df = pd.read_parquet(path or DATA)
    df = df.set_index("ts_utc").sort_index()
    df.index = df.index.tz_convert("Europe/Berlin")
    return df


def _day(df: pd.DataFrame, day: str) -> pd.DataFrame:
    d = pd.Timestamp(day).tz_localize("Europe/Berlin")
    return df.loc[d: d + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)]


def data_quality(day: str, df: pd.DataFrame | None = None) -> dict:
    """Checks one day: missing or duplicate hours, missing values, night solar, impossible values."""
    df = load() if df is None else df
    x = _day(df, day)
    d0 = pd.Timestamp(day).tz_localize("Europe/Berlin")
    expected = len(pd.date_range(d0, d0 + pd.Timedelta(days=1), freq="h", inclusive="left"))
    night = x.between_time("00:00", "03:00")
    flags = []
    if len(x) < expected:
        flags.append(f"missing_hours:{expected - len(x)}")
    if x.index.duplicated().any():
        flags.append("duplicate_timestamps")
    nmiss = int(x.isna().sum().sum())
    if nmiss:
        flags.append(f"missing_values:{nmiss}")
    if len(night) and (night["solar"] > 500).any():
        flags.append("solar_at_night")
    if (x["load"] < 15000).any() or (x["load"] > 90000).any():
        flags.append("load_out_of_range")
    if (x["price"].abs() > 1000).any():
        flags.append("price_out_of_range")
    return {"day": day, "hours": int(len(x)), "expected_hours": int(expected), "flags": flags}


def daily_summary(day: str, df: pd.DataFrame | None = None) -> dict:
    df = load() if df is None else df
    x = _day(df, day)
    if x.empty:
        return {"day": day, "error": "no data"}
    res = x["load"] - x["wind_onshore"] - x["wind_offshore"] - x["solar"]
    return {
        "day": day,
        "price_mean": round(float(x["price"].mean()), 2),
        "price_min": round(float(x["price"].min()), 2),
        "price_max": round(float(x["price"].max()), 2),
        "negative_price_hours": int((x["price"] < 0).sum()),
        "wind_mean_mw": round(float((x["wind_onshore"] + x["wind_offshore"]).mean()), 0),
        "solar_peak_mw": round(float(x["solar"].max()), 0),
        "residual_load_mean_mw": round(float(res.mean()), 0),
        "price_at_max_residual_load": round(float(x["price"][res.idxmax()]), 2),
    }


def price_anomalies(day: str, df: pd.DataFrame | None = None, window_days: int = 28, k: float = 4.0) -> dict:
    """Hours whose price differs from the median of the same hour in the previous window_days
    by more than k times the robust spread (MAD) of that hour."""
    df = load() if df is None else df
    d0 = pd.Timestamp(day).tz_localize("Europe/Berlin")
    hist = df.loc[d0 - pd.Timedelta(days=window_days): d0 - pd.Timedelta(seconds=1)]
    x = _day(df, day)
    out = []
    for ts, row in x.iterrows():
        same = hist[hist.index.hour == ts.hour]["price"]
        if len(same) < 10:
            continue
        med = float(same.median())
        mad = float((same - med).abs().median()) * 1.4826
        mad = max(mad, 5.0)
        z = (row["price"] - med) / mad
        if abs(z) > k:
            out.append({"hour": int(ts.hour), "price": round(float(row["price"]), 2),
                        "typical": round(med, 2), "z": round(float(z), 1)})
    return {"day": day, "n_anomalies": len(out), "anomalies": out}


def compare_days(day_a: str, day_b: str, df: pd.DataFrame | None = None) -> dict:
    a, b = daily_summary(day_a, df), daily_summary(day_b, df)
    return {"a": a, "b": b, "price_mean_diff": round(a["price_mean"] - b["price_mean"], 2),
            "wind_mean_diff_mw": round(a["wind_mean_mw"] - b["wind_mean_mw"], 0)}


def wind_price_slope(start: str, end: str, df: pd.DataFrame | None = None) -> dict:
    """OLS slope of price on residual load (EUR/MWh per GW) between two dates."""
    df = load() if df is None else df
    s, e = pd.Timestamp(start).tz_localize("Europe/Berlin"), pd.Timestamp(end).tz_localize("Europe/Berlin")
    x = df.loc[s:e].dropna()
    res = (x["load"] - x["wind_onshore"] - x["wind_offshore"] - x["solar"]) / 1000.0
    if len(x) < 48:
        return {"error": "too few hours"}
    b, a = np.polyfit(res.values, x["price"].values, 1)
    r = float(np.corrcoef(res.values, x["price"].values)[0, 1])
    return {"start": start, "end": end, "hours": int(len(x)), "eur_per_mwh_per_gw": round(float(b), 2),
            "correlation": round(r, 2)}


KB = Path(__file__).resolve().parents[2] / "kb"


@lru_cache(maxsize=1)
def _kb_index():
    from sklearn.feature_extraction.text import TfidfVectorizer
    docs = [(f.stem, f.read_text(encoding="utf-8")) for f in sorted(KB.glob("*.md"))]
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    return docs, vec, vec.fit_transform([t for _, t in docs])


def search_kb(query: str, df=None, k: int = 2, min_score: float = 0.12) -> dict:
    """Passages from the operations knowledge base by TF-IDF cosine similarity. Nothing is generated."""
    docs, vec, mat = _kb_index()
    sims = (mat @ vec.transform([query]).T).toarray().ravel()
    order = np.argsort(-sims)[:k]
    hits = [{"source": docs[i][0], "score": round(float(sims[i]), 2), "text": docs[i][1].strip()}
            for i in order if sims[i] >= min_score]
    return {"query": query, "hits": hits, "found": bool(hits)}


def draft_notice(day: str, df: pd.DataFrame | None = None) -> dict:
    """Draft customer notice for a day, built only from tool results; needs_approval is always true."""
    s, a = daily_summary(day, df), price_anomalies(day, df)
    if "error" in s:
        return {"day": day, "error": "no data"}
    text = (f"On {day} the German day-ahead price ranged from {s['price_min']} to {s['price_max']} EUR/MWh "
            f"(mean {s['price_mean']}). {a['n_anomalies']} hours were unusual compared with the previous 28 days. "
            "These are market prices; this message makes no statement about your tariff.")
    return {"day": day, "n_anomalies": a["n_anomalies"], "text": text, "needs_approval": True}


REGISTRY = {
    "search_kb": {"fn": search_kb, "args": {"query": "text"},
                  "doc": "Passages from the operations knowledge base (definitions, alert rules, fault handling)."},
    "draft_notice": {"fn": draft_notice, "args": {"day": "YYYY-MM-DD"},
                     "doc": "Draft customer notice for a day; always needs human approval."},
    "data_quality": {"fn": data_quality, "args": {"day": "YYYY-MM-DD"},
                     "doc": "Completeness and plausibility checks for one day."},
    "daily_summary": {"fn": daily_summary, "args": {"day": "YYYY-MM-DD"},
                      "doc": "Price, wind, solar and residual load of one day."},
    "price_anomalies": {"fn": price_anomalies, "args": {"day": "YYYY-MM-DD"},
                        "doc": "Hours with unusual prices against the same hour in the previous 28 days."},
    "compare_days": {"fn": compare_days, "args": {"day_a": "YYYY-MM-DD", "day_b": "YYYY-MM-DD"},
                     "doc": "Two daily summaries side by side."},
    "wind_price_slope": {"fn": wind_price_slope, "args": {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"},
                         "doc": "Price change per GW of residual load over a period."},
}
