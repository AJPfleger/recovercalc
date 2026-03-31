def run_today():
    from .metrics import (
        _hr_zone_frac,
        _trimp_from_samples,
        add_ctl_atl,
        add_progression_metrics,
        recent_pace_km_per_min,
        recent_pace_min_per_km,
    )

    from .decision import decide_today, next_week_targets

    from .builders import (
        HR_ZONES,
        hr_target,
        build_easy_session,
        build_long_session,
        build_quality_session,
        allocate_sessions,
        schedule_week,
        print_week,
    )

    from .io_fit import load_history, export_workout_from_template_like_structure

    from .plots import plot_load, plot_progression

    from pathlib import Path
    from dataclasses import dataclass, asdict
    import math
    import pandas as pd
    import numpy as np
    import fitdecode

    LOCAL_TZ = "Europe/Paris"

    @dataclass
    class RunSummary:
        file: str
        start_time: pd.Timestamp | None
        duration_s: float
        distance_m: float
        avg_hr: float | None
        gain_m: float
        avg_pace_s_per_km: float | None
        trimp: float
        week: str
        z1: float
        z2: float
        z3: float
        z4: float
        z5: float

    # usage
    activities, runs, weekly = load_history("data", history_days=365)
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

    # test
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
    pace_km_per_min = recent_pace_km_per_min(runs)
    plan = allocate_sessions(targets["target_km"], targets["runs"], pace_km_per_min)

    print(targets)
    print({"pace_km_per_min": pace_km_per_min})
    print(plan)
    print("sum_km =", round(sum(s["km"] for s in plan["sessions"]), 2))

    def recent_pace_min_per_km(runs: pd.DataFrame, lookback: int = 10):
        x = runs.sort_values("start_time").tail(lookback)
        x = x[(x["distance_m"] > 0) & (x["duration_s"] > 0)]
        if x.empty:
            return 6.0
        km = x["distance_m"].sum() / 1000.0
        minutes = x["duration_s"].sum() / 60.0
        return float(minutes / km)

    preferred_days = ["Mon", "Wed", "Fri"]  # change weekly
    pace = recent_pace_min_per_km(runs)
    week = schedule_week(plan, pace, daily["state"].iloc[-1], preferred_days)
    print_week(week)

    from fit_tool.fit_file_builder import FitFileBuilder
    from fit_tool.profile.messages.workout_message import WorkoutMessage
    from fit_tool.profile.messages.workout_step_message import WorkoutStepMessage

    from fit_tool.profile.profile_type import (
        WorkoutStepDuration,
        WorkoutStepTarget,
        Intensity,
    )

    pace = 1.0 / recent_pace_km_per_min(runs)

    if today_type == "REST":
        print(f"REST day: TSB={daily['tsb'].iloc[-1]:.1f}, no workout exported.")
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
    print_week({"Today": workout})

    export_workout_from_template_like_structure(
        workout=workout,
        filename="today.fit",
        workout_name="recovercalc " + workout_name,
    )
    pass
