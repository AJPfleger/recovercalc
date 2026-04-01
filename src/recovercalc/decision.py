import pandas as pd
import numpy as np

from .config import (
    LOCAL_TZ,
    EASY_TSB_MIN,
    LONG_TSB_MIN,
    QUALITY_TSB_MIN,
    HIGH_LOAD_TRIMP,
    VERY_HIGH_LOAD_TRIMP,
    MIN_DAYS_BETWEEN_RUNS,
    MIN_DAYS_AFTER_LONG,
    MIN_DAYS_AFTER_QUALITY,
    QUALITY_GAP_DAYS,
    LONG_GAP_DAYS,
    LONG_DURATION_MIN,
    LONG_TRIMP_MIN,
    QUALITY_TRIMP_MIN,
    QUALITY_Z4Z5_FRAC_MIN,
    QUALITY_Z3_FRAC_MIN,
)


def _local_day(ts: pd.Series) -> pd.Series:
    return pd.to_datetime(ts, utc=True).dt.tz_convert(LOCAL_TZ).dt.floor("D")


def _today_local(today: pd.Timestamp | None = None) -> pd.Timestamp:
    if today is None:
        return pd.Timestamp.now("UTC").tz_convert(LOCAL_TZ).floor("D")

    t = pd.Timestamp(today)
    if t.tzinfo is None:
        return t.tz_localize(LOCAL_TZ).floor("D")

    return t.tz_convert(LOCAL_TZ).floor("D")


def _classify_run(row: pd.Series) -> str:
    duration_min = float(row.get("duration_s", 0.0)) / 60.0
    trimp = float(row.get("trimp", 0.0))
    z3 = float(row.get("z3", 0.0))
    z4 = float(row.get("z4", 0.0))
    z5 = float(row.get("z5", 0.0))
    z4z5 = z4 + z5

    if duration_min >= LONG_DURATION_MIN and trimp >= LONG_TRIMP_MIN:
        return "LONG"

    if trimp >= QUALITY_TRIMP_MIN and (
        z4z5 >= QUALITY_Z4Z5_FRAC_MIN or z3 >= QUALITY_Z3_FRAC_MIN
    ):
        return "QUALITY"

    return "EASY"


def decide_today(
    daily: pd.DataFrame,
    runs: pd.DataFrame,
    activities: pd.DataFrame,
    today: pd.Timestamp | None = None,
    quality_gap_days: int = QUALITY_GAP_DAYS,
    long_gap_days: int = LONG_GAP_DAYS,
) -> str:
    today_local = _today_local(today)

    daily_local = daily.copy()
    daily_local["local_day"] = daily_local.index.tz_convert(LOCAL_TZ).floor("D")
    daily_local = daily_local.set_index("local_day", drop=False)

    tsb_series = daily_local["tsb"].shift(1)
    trimp_series = daily_local["trimp"]

    tsb_today = float(tsb_series.loc[tsb_series.index <= today_local].iloc[-1])
    yesterday_trimp_all = float(
        trimp_series.loc[trimp_series.index < today_local].iloc[-1]
    )

    # Previous day was too exhausting
    if yesterday_trimp_all >= VERY_HIGH_LOAD_TRIMP:
        return "REST"

    # Too fatigued even for an easy run
    if tsb_today < EASY_TSB_MIN:
        return "REST"

    # Not fit enough for a not-easy run
    if tsb_today < min(QUALITY_TSB_MIN, LONG_TSB_MIN):
        return "EASY"

    # No runs in the history at all, better start easy
    if runs.empty:
        return "EASY"

    # Yesterday was too hard for more than easy
    if yesterday_trimp_all >= HIGH_LOAD_TRIMP:
        return "EASY"

    runs_local = runs.copy()
    runs_local["local_day"] = _local_day(runs_local["start_time"])
    runs_local = runs_local.sort_values("start_time").reset_index(drop=True)

    last_run = runs_local.iloc[-1]
    last_run_day = pd.Timestamp(last_run["local_day"])
    days_since_run = int((today_local - last_run_day).days)

    # Usually prevents to have multiple runs on the same day
    if days_since_run < MIN_DAYS_BETWEEN_RUNS:
        return "REST"

    run_types = runs_local.apply(_classify_run, axis=1)
    long_days = runs_local.loc[run_types == "LONG", "local_day"]
    quality_days = runs_local.loc[run_types == "QUALITY", "local_day"]

    days_since_long = (
        999
        if long_days.empty
        else int((today_local - pd.Timestamp(long_days.iloc[-1])).days)
    )
    days_since_quality = (
        999
        if quality_days.empty
        else int((today_local - pd.Timestamp(quality_days.iloc[-1])).days)
    )
    last_run_type = _classify_run(last_run)

    if last_run_type == "LONG" and days_since_run < MIN_DAYS_AFTER_LONG:
        return "EASY"

    if last_run_type == "QUALITY" and days_since_run < MIN_DAYS_AFTER_QUALITY:
        return "EASY"

    if (
        days_since_quality >= quality_gap_days
        and tsb_today >= QUALITY_TSB_MIN
        and days_since_run >= MIN_DAYS_AFTER_QUALITY
        and days_since_long >= MIN_DAYS_AFTER_LONG
    ):
        return "QUALITY"

    if (
        days_since_long >= long_gap_days
        and tsb_today >= LONG_TSB_MIN
        and days_since_run >= MIN_DAYS_AFTER_LONG
        and days_since_quality >= MIN_DAYS_AFTER_QUALITY
    ):
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


def forecast_next_training_day(
    daily, runs, activities, max_days=14, ctl_tau=42.0, atl_tau=7.0
):
    sim = daily.copy()
    last = sim.iloc[-1].copy()
    for d in range(1, max_days + 1):
        last["ctl"] = last["ctl"] + (0.0 - last["ctl"]) / ctl_tau
        last["atl"] = last["atl"] + (0.0 - last["atl"]) / atl_tau
        last["tsb"] = last["ctl"] - last["atl"]
        sim.loc[sim.index[-1] + pd.Timedelta(days=d)] = {
            "trimp": 0.0,
            "ctl": last["ctl"],
            "atl": last["atl"],
            "tsb": last["tsb"],
        }
        if decide_today(sim, runs, activities, today=sim.index[-1]) != "REST":
            return (
                d,
                decide_today(sim, runs, activities, today=sim.index[-1]),
                float(last["tsb"]),
            )
    return None, "REST", float(last["tsb"])
