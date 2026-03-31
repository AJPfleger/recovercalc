import pandas as pd

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
