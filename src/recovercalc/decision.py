import pandas as pd
import numpy as np

from .config import LOCAL_TZ


def decide_today(
    daily: pd.DataFrame,
    runs: pd.DataFrame,
    activities: pd.DataFrame,
    today: pd.Timestamp | None = None,
    quality_gap_days: int = 10,
    long_gap_days: int = 7,
):
    today = (
        pd.Timestamp.now("UTC").tz_convert(LOCAL_TZ).floor("D")
        if today is None
        else (
            pd.Timestamp(today).tz_localize(LOCAL_TZ)
            if pd.Timestamp(today).tzinfo is None
            else pd.Timestamp(today).tz_convert(LOCAL_TZ).floor("D")
        )
    )
    run_days = (
        pd.to_datetime(runs["start_time"], utc=True)
        .dt.tz_convert(LOCAL_TZ)
        .dt.floor("D")
    )
    last_run_day = run_days.max() if not runs.empty else None
    tsb = float(daily.loc[daily.index <= today, "tsb"].iloc[-1])
    days_since_run = 999 if pd.isna(last_run_day) else int((today - last_run_day).days)
    run_dur = (
        runs.groupby(run_days)["duration_s"].sum()
        if not runs.empty
        else pd.Series(dtype=float)
    )
    quality_days = (
        run_days[run_dur.reindex(run_days).values >= 35 * 60]
        if not runs.empty
        else pd.Series(dtype="datetime64[ns, UTC]")
    )
    long_days = (
        run_days[run_dur.reindex(run_days).values >= 60 * 60]
        if not runs.empty
        else pd.Series(dtype="datetime64[ns, UTC]")
    )
    days_since_quality = (
        999 if len(quality_days) == 0 else int((today - quality_days.max()).days)
    )
    days_since_long = (
        999 if len(long_days) == 0 else int((today - long_days.max()).days)
    )
    if tsb < -10 or days_since_run < 1:
        return "REST"
    if days_since_quality >= quality_gap_days and tsb > -5 and days_since_run >= 2:
        return "QUALITY"
    if days_since_long >= long_gap_days and days_since_run >= 1:
        return "LONG"
    return "EASY"


def next_week_targets(
    daily: pd.DataFrame,
    weekly: pd.DataFrame,
    max_weekly_km: float = 25.0,
    min_runs: int = 3,
    max_runs: int = 5,
):
    w = weekly.copy().sort_values("week")
    last = w.iloc[-1]
    recent_km = float(w["distance_km"].tail(min(4, len(w))).mean())
    recent_trimp = float(w["trimp"].tail(min(4, len(w))).mean())

    ctl = float(daily["ctl"].iloc[-1])
    tsb = float(daily["tsb"].iloc[-1])

    if tsb < -10:
        km_target = recent_km * 0.90
        trimp_target = recent_trimp * 0.90
    elif tsb > 5:
        km_target = recent_km * 1.08
        trimp_target = recent_trimp * 1.08
    else:
        km_target = recent_km * 1.04
        trimp_target = recent_trimp * 1.04

    ctl_target = ctl * 7.0
    km_target = min(km_target, float(last["distance_km"]) + 2.5, max_weekly_km)
    trimp_target = min(trimp_target, float(last["trimp"]) * 1.08, ctl_target * 1.10)

    runs = int(
        np.clip(round(float(w["runs"].tail(min(4, len(w))).mean())), min_runs, max_runs)
    )

    return {
        "target_km": float(km_target),
        "target_trimp": float(trimp_target),
        "runs": runs,
    }
