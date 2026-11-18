"""Load and audit the fixed InfoDengue Rio de Janeiro weekly case snapshot."""
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
RAW_CSV = HERE / "data" / "raw" / "infodengue_rio_2010_2025_snapshot_20261118.csv"
TARGET_COLUMN = "casos"  # reported cases; deliberately not the nowcast estimate `casos_est`
SERIES_LABEL = "Rio de Janeiro, Brazil (InfoDengue reported dengue cases)"


def load_raw() -> pd.DataFrame:
    source = pd.read_csv(RAW_CSV)
    required = {"data_iniSE", TARGET_COLUMN, "SE", "municipio_nome", "versao_modelo"}
    missing = required.difference(source.columns)
    if missing:
        raise ValueError(f"InfoDengue snapshot is missing columns: {sorted(missing)}")
    if set(source["municipio_nome"].dropna().unique()) != {"Rio de Janeiro"}:
        raise ValueError("Snapshot contains unexpected municipality names")
    out = pd.DataFrame({
        "date": pd.to_datetime(source["data_iniSE"], errors="raise"),
        "cases": pd.to_numeric(source[TARGET_COLUMN], errors="raise"),
    }).sort_values("date").reset_index(drop=True)
    if out["date"].duplicated().any():
        raise ValueError("Duplicate epidemiological week starts found; stop and audit")
    if out["cases"].isna().any() or (out["cases"] < 0).any():
        raise ValueError("Missing or negative reported case counts found; stop and audit")
    gaps = out["date"].diff().dropna().dt.days
    if not gaps.eq(7).all():
        raise ValueError(f"Series is not gapless weekly data: {gaps.value_counts().sort_index().to_dict()}")
    return out


def audit() -> dict:
    df = load_raw()
    return {
        "rows": len(df),
        "date_start": df["date"].min().date().isoformat(),
        "date_end": df["date"].max().date().isoformat(),
        "duplicate_dates": int(df["date"].duplicated().sum()),
        "missing_cases": int(df["cases"].isna().sum()),
        "negative_cases": int((df["cases"] < 0).sum()),
        "gap_days": df["date"].diff().dropna().dt.days.value_counts().sort_index().to_dict(),
        "minimum": int(df["cases"].min()),
        "median": float(df["cases"].median()),
        "maximum": int(df["cases"].max()),
    }


if __name__ == "__main__":
    import json
    print(json.dumps(audit(), indent=2))
