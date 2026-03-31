def run_today():
    from .metrics import (
        _hr_zone_frac,
        _trimp_from_samples,
        add_ctl_atl,
        add_progression_metrics,
        recent_pace_km_per_min,
        recent_pace_min_per_km,
    )

    from .decision import decide_today

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

    from pathlib import Path
    from dataclasses import dataclass, asdict
    import math
    import pandas as pd
    import numpy as np
    import fitdecode

    REST_HR, MAX_HR = 45, 190  # adjust
    HISTORY_DAYS = 365  # or None
    MIN_SESSION_MIN = 20.0
    MIN_SESSION_KM = 3.2
    MIN_LONG_KM = 6.0
    MIN_QUALITY_KM = 4.0
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

    def plot_load(daily: pd.DataFrame):
        import matplotlib.pyplot as plt

        ax = daily[["trimp", "ctl", "atl", "tsb"]].plot(
            figsize=(10, 5), secondary_y=["tsb"]
        )
        ax.set_ylabel("TRIMP / load")
        ax.right_ax.set_ylabel("TSB")
        plt.tight_layout()
        plt.show()

    def plot_progression(daily: pd.DataFrame, weekly: pd.DataFrame):
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 5))
        plt.plot(daily.index, daily["ctl_ramp_7"], label="ctl_ramp_7")
        plt.plot(daily.index, daily["ctl_ramp_28"], label="ctl_ramp_28")
        plt.axhline(0.0, linewidth=1)
        plt.ylabel("CTL ramp")
        plt.legend()
        plt.tight_layout()
        plt.show()

        if "distance_km" in weekly.columns and "dist_ramp" in weekly.columns:
            plt.figure(figsize=(10, 5))
            plt.plot(
                pd.to_datetime(weekly["week"]), weekly["distance_km"], label="distance_km"
            )
            plt.plot(pd.to_datetime(weekly["week"]), weekly["dist_ramp"], label="dist_ramp")
            plt.axhline(0.0, linewidth=1)
            plt.legend()
            plt.tight_layout()
            plt.show()

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

    # test
    targets = next_week_targets(daily, weekly)
    print(targets)

    # usage
    targets = next_week_targets(daily, weekly)
    pace_km_per_min = recent_pace_km_per_min(runs)
    plan = allocate_sessions(targets["target_km"], targets["runs"], pace_km_per_min)

    print(targets)
    print({"pace_km_per_min": pace_km_per_min})
    print(plan)
    print("sum_km =", round(sum(s["km"] for s in plan["sessions"]), 2))

    def build_quality_session(target_km: float, state: str = "normal"):
        if target_km < 5.0:
            return {
                "kind": "easy",
                "steps": [
                    {"type": "warmup", "min": 10},
                    {"type": "easy", "min": 20},
                    {"type": "cooldown", "min": 5},
                ],
            }
        if state == "fatigued":
            return {
                "kind": "tempo",
                "steps": [
                    {"type": "warmup", "min": 10},
                    {"type": "steady", "repeats": 2, "on_min": 6, "off_min": 3},
                    {"type": "cooldown", "min": 10},
                ],
            }
        if target_km < 6.5:
            return {
                "kind": "interval",
                "steps": [
                    {"type": "warmup", "min": 10},
                    {"type": "interval", "repeats": 6, "on_min": 2, "off_min": 2},
                    {"type": "cooldown", "min": 10},
                ],
            }
        return {
            "kind": "tempo",
            "steps": [
                {"type": "warmup", "min": 12},
                {"type": "steady", "repeats": 3, "on_min": 6, "off_min": 3},
                {"type": "cooldown", "min": 10},
            ],
        }

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
    targets = next_week_targets(daily, weekly)

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
