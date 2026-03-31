import numpy as np
import pandas as pd
from .config import REST_HR, MAX_HR, HR_ZONES


def _hr_zone_frac(hr: np.ndarray) -> dict[str, float]:
    if hr.size == 0:
        return {z.lower(): 0.0 for z in HR_ZONES}

    hrr = np.clip((hr - REST_HR) / (MAX_HR - REST_HR), 0.0, 1.5)

    zones = {}
    items = list(HR_ZONES.items())
    for i, (name, (lo, hi)) in enumerate(items):
        if i == len(items) - 1:
            mask = (hrr >= lo) & (hrr <= hi)
        else:
            mask = (hrr >= lo) & (hrr < hi)
        zones[name.lower()] = float(mask.mean())

    return zones


def _trimp_from_samples(hr: np.ndarray, dt_s: np.ndarray) -> float:
    if hr.size == 0 or dt_s.size == 0:
        return 0.0
    hrr = np.clip((hr - REST_HR) / (MAX_HR - REST_HR), 0, 1.2)
    y = 0.64 * np.exp(1.92 * hrr)  # Banister male; replace if needed
    return float(np.sum((dt_s / 60.0) * hrr * y))


def extend_daily_to_today(daily: pd.DataFrame, today=None, ctl_tau=42.0, atl_tau=7.0):
    today = pd.Timestamp.now("UTC").floor("D") if today is None else pd.Timestamp(today)
    out = daily.copy()
    last_day = out.index.max().floor("D")
    while last_day < today:
        next_day = last_day + pd.Timedelta(days=1)
        prev = out.iloc[-1]
        ctl = prev["ctl"] + (0.0 - prev["ctl"]) / ctl_tau
        atl = prev["atl"] + (0.0 - prev["atl"]) / atl_tau
        out.loc[next_day] = {"trimp": 0.0, "ctl": ctl, "atl": atl, "tsb": ctl - atl}
        last_day = next_day.floor("D")
    return out


def add_ctl_atl(activities: pd.DataFrame, ctl_tau: float = 42.0, atl_tau: float = 7.0):
    df = activities.copy()
    df["date"] = pd.to_datetime(df["start_time"], utc=True).dt.floor("D")
    daily = df.groupby("date").agg(trimp=("trimp", "sum"))
    idx = pd.date_range(daily.index.min(), daily.index.max(), freq="D", tz="UTC")
    daily = daily.reindex(idx, fill_value=0.0)
    ctl = np.zeros(len(daily))
    atl = np.zeros(len(daily))
    init = (
        daily["trimp"].iloc[:42].mean() if len(daily) >= 42 else daily["trimp"].mean()
    )
    ctl[0] = init
    atl[0] = init
    for i in range(1, len(daily)):
        ctl[i] = ctl[i - 1] + (daily["trimp"].iloc[i] - ctl[i - 1]) / ctl_tau
        atl[i] = atl[i - 1] + (daily["trimp"].iloc[i] - atl[i - 1]) / atl_tau
    daily["ctl"], daily["atl"], daily["tsb"] = ctl, atl, ctl - atl

    return extend_daily_to_today(daily)


def add_progression_metrics(daily: pd.DataFrame, weekly: pd.DataFrame):
    daily = daily.copy()
    weekly = weekly.copy()

    daily["ctl_ramp_7"] = daily["ctl"].diff(7) / 7.0
    daily["ctl_ramp_28"] = daily["ctl"].diff(28) / 28.0
    daily["fatigue_flag"] = daily["tsb"] < -10.0
    daily["fresh_flag"] = daily["tsb"] > 5.0

    def classify_state(tsb: float) -> str:
        if pd.isna(tsb):
            return "unknown"
        if tsb > 5.0:
            return "fresh"
        if tsb < -10.0:
            return "fatigued"
        return "normal"

    daily["state"] = daily["tsb"].apply(classify_state)

    if "distance_km" in weekly.columns:
        weekly["dist_ramp"] = weekly["distance_km"].pct_change()
        weekly["dist_change_km"] = weekly["distance_km"].diff()
    if "trimp" in weekly.columns:
        weekly["trimp_ramp"] = weekly["trimp"].pct_change()
        weekly["trimp_change"] = weekly["trimp"].diff()

    return daily, weekly


def recent_pace_km_per_min(runs: pd.DataFrame, lookback: int = 10) -> float:
    x = runs.sort_values("start_time").tail(lookback).copy()
    x = x[(x["distance_m"] > 0) & (x["duration_s"] > 0)]
    if x.empty:
        return 0.10  # 6:00 min/km fallback
    return float((x["distance_m"].sum() / 1000.0) / (x["duration_s"].sum() / 60.0))


def recent_pace_min_per_km(runs: pd.DataFrame, lookback: int = 10) -> float:
    return 1.0 / recent_pace_km_per_min(runs, lookback)
