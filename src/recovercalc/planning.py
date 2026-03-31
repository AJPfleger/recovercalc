from .metrics import (
    _hr_zone_frac,
    _trimp_from_samples,
    add_ctl_atl,
    add_progression_metrics,
    recent_pace_km_per_min,
    recent_pace_min_per_km,
)

from .decision import decide_today, next_week_targets, forecast_next_training_day

from .builders import (
    HR_ZONES,
    hr_target,
    build_easy_session,
    build_long_session,
    build_quality_session,
)

from .io_fit import load_history, export_workout_from_template_like_structure

from .plots import plot_load, plot_progression, plot_training_overview

import pandas as pd

LOCAL_TZ = "Europe/Paris"


def run_today(history_days: int = 365):
    activities, runs, weekly = load_history(
        "data/activity_fits", history_days=history_days
    )

    if activities.empty:
        print("ERROR: no activities found → cannot compute training")
        raise SystemExit(1)

    span = (activities["start_time"].max() - activities["start_time"].min()).days

    if span < 42:
        print("WARNING: <42 days activity history → CTL unstable")

    daily = add_ctl_atl(activities)
    daily, weekly = add_progression_metrics(daily, weekly)

    plot_load(daily)
    plot_progression(daily, weekly)

    print(
        daily[
            [
                "trimp",
                "ctl",
                "atl",
                "tsb",
                "ctl_ramp_7",
                "ctl_ramp_28",
                "fatigue_flag",
                "state",
            ]
        ].tail(10)
    )
    print(weekly[["week", "distance_km", "trimp", "dist_ramp", "trimp_ramp"]].tail(10))

    today_type = decide_today(daily, runs, activities)
    print("today_type =", today_type)

    print("TSB:", daily["tsb"].iloc[-1])
    print(
        "days_since_last_run:",
        (
            pd.Timestamp.now("UTC").tz_convert(LOCAL_TZ).floor("D")
            - pd.to_datetime(runs["start_time"], utc=True)
            .dt.tz_convert(LOCAL_TZ)
            .dt.floor("D")
            .max()
        ).days,
    )
    print("decision:", today_type)

    targets = next_week_targets(daily, weekly)
    print(targets)

    pace = recent_pace_min_per_km(runs)

    if today_type == "REST":
        print(f"REST day: TSB={daily['tsb'].iloc[-1]:.1f}, no workout exported.")
        days, next_type, next_tsb = forecast_next_training_day(daily, runs, activities)
        print(
            f"REST today. Next training in {days} day(s): {next_type} (predicted TSB={next_tsb:.1f})"
        )
        raise SystemExit(0)
    elif today_type == "EASY":
        target_km = max(3.2, targets["target_km"] / max(targets["runs"], 1))
        workout = build_easy_session(target_km=target_km, pace_min_per_km=pace)
        workout_name = "Easy Run"
    elif today_type == "LONG":
        target_km = max(6.0, targets["target_km"] * 0.35)
        workout = build_long_session(target_km=target_km, pace_min_per_km=pace)
        workout_name = "Long Run"
    elif today_type == "QUALITY":
        target_km = max(4.0, targets["target_km"] * 0.20)
        workout = build_quality_session(
            target_km=target_km, pace_min_per_km=pace, state=daily["state"].iloc[-1]
        )
        workout_name = "Quality Run"
    else:
        raise ValueError(f"unknown today_type: {today_type}")

    print(f"today_type={today_type}, workout_name={workout_name}")
    print(workout)

    export_workout_from_template_like_structure(
        workout=workout,
        filename="today.fit",
        workout_name="recovercalc " + workout_name,
    )

    plot_training_overview(daily, weekly)

    pass
