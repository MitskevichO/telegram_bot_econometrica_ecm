from __future__ import annotations

import pandas as pd
from io import StringIO

def session_df(session: dict) -> pd.DataFrame:
    return pd.read_json(StringIO(session["df_json"]), orient="split")

def pick_yx_columns(session: dict) -> tuple[str, str]:
    if session.get("roles") and session["roles"].get("y") and session["roles"].get("x"):
        y = session["roles"]["y"]
        x = session["roles"]["x"]
        print(f"[DEBUG] pick_yx_columns: Y={y}, X={x}")
        return y, x
    v = session["validation"]
    cols = v["numeric_cols"]
    print(f"[DEBUG] pick_yx_columns (fallback): Y={cols[0]}, X={cols[1]}")
    return cols[0], cols[1]