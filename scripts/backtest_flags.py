"""How often does the daily report raise a flag over 2024-2025, and on which days?"""
import json, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eoa import tools as T

df = T.load()
days = pd.date_range("2024-01-29", "2025-12-30", freq="D")
rows = []
for d in days:
    ds = d.strftime("%Y-%m-%d")
    q = T.data_quality(ds, df); a = T.price_anomalies(ds, df)
    rows.append({"day": ds, "quality_flags": len(q["flags"]), "n_anomalies": a["n_anomalies"],
                 "price_mean": T.daily_summary(ds, df)["price_mean"]})
r = pd.DataFrame(rows)
top = r.sort_values("n_anomalies", ascending=False).head(10)
out = {"days": int(len(r)), "days_with_quality_flag": int((r.quality_flags > 0).sum()),
       "days_with_anomaly": int((r.n_anomalies > 0).sum()), "days_with_8plus_anomalies": int((r.n_anomalies >= 8).sum()),
       "top_days": top.to_dict("records")}
Path("results").mkdir(exist_ok=True)
json.dump(out, open("results/flag_backtest.json", "w"), indent=1)
print(json.dumps(out, indent=1))
