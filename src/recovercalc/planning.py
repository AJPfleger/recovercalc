def run_today():
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


    def _hr_zone_frac(hr: np.ndarray) -> dict[str, float]:
        if hr.size == 0:
            return {f"z{i}": 0.0 for i in range(1, 6)}
        hrr = (hr - REST_HR) / (MAX_HR - REST_HR)
        bins = [0.60, 0.70, 0.80, 0.90]
        idx = np.digitize(hrr, bins, right=False)
        return {f"z{i+1}": float((idx == i).mean()) for i in range(5)}


    def _trimp_from_samples(hr: np.ndarray, dt_s: np.ndarray) -> float:
        if hr.size == 0 or dt_s.size == 0:
            return 0.0
        hrr = np.clip((hr - REST_HR) / (MAX_HR - REST_HR), 0, 1.2)
        y = 0.64 * np.exp(1.92 * hrr)  # Banister male; replace if needed
        return float(np.sum((dt_s / 60.0) * hrr * y))


    # 1) parser: keep all sports, mark running separately
    @dataclass
    class ActivitySummary:
        file: str
        sport: str | None
        sub_sport: str | None
        is_run: bool
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


    def parse_activity_fit(path: str | Path) -> ActivitySummary | None:
        recs, ses = [], {}
        with fitdecode.FitReader(str(path)) as fit:
            for frame in fit:
                if not isinstance(frame, fitdecode.FitDataMessage):
                    continue
                vals = {f.name: f.value for f in frame.fields}
                if frame.name == "session":
                    ses = vals
                elif frame.name == "record":
                    recs.append(vals)
        if not ses:
            return None
        sport, sub_sport = ses.get("sport"), ses.get("sub_sport")
        is_run = sport == "running"
        df = pd.DataFrame(recs)
        if "timestamp" in df:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        if "enhanced_altitude" not in df and "altitude" in df:
            df["enhanced_altitude"] = df["altitude"]
        duration_s = float(ses.get("total_timer_time", 0) or 0)
        distance_m = float(ses.get("total_distance", 0) or 0)
        avg_hr = (
            float(ses["avg_heart_rate"]) if ses.get("avg_heart_rate") is not None else None
        )
        start_time = (
            pd.to_datetime(ses["start_time"], utc=True)
            if ses.get("start_time") is not None
            else (df["timestamp"].iloc[0] if "timestamp" in df and not df.empty else None)
        )
        gain_m = float(ses.get("total_ascent", 0) or 0)
        if gain_m == 0 and "enhanced_altitude" in df and len(df) > 1:
            gain_m = float(df["enhanced_altitude"].diff().clip(lower=0).sum(skipna=True))
        avg_pace = (
            (duration_s / (distance_m / 1000.0)) if is_run and distance_m > 0 else None
        )
        hr = (
            df["heart_rate"].dropna().to_numpy(dtype=float)
            if "heart_rate" in df
            else np.array([])
        )
        if "timestamp" in df and len(df) > 1:
            dt_s = (
                df["timestamp"]
                .diff()
                .dt.total_seconds()
                .fillna(0)
                .clip(lower=0, upper=30)
                .to_numpy(dtype=float)
            )
            dt_s = dt_s[1:] if hr.size and len(dt_s) == len(df) else dt_s
            if hr.size and len(hr) == len(df):
                hr = hr[1:]
        else:
            dt_s = np.full(max(len(hr), 1), duration_s / max(len(hr), 1), dtype=float)
        zones = _hr_zone_frac(hr) if hr.size else {f"z{i}": 0.0 for i in range(1, 6)}
        trimp = _trimp_from_samples(hr, dt_s[: len(hr)]) if hr.size else 0.0
        week = (
            (
                start_time.tz_convert(None)
                if getattr(start_time, "tzinfo", None)
                else start_time
            )
            .to_period("W-MON")
            .start_time.strftime("%Y-%m-%d")
            if start_time is not None
            else "unknown"
        )
        return ActivitySummary(
            str(path),
            sport,
            sub_sport,
            is_run,
            start_time,
            duration_s,
            distance_m,
            avg_hr,
            gain_m,
            avg_pace,
            trimp,
            week,
            **zones,
        )


    # 2) loader: return all activities + runs + run-weekly
    def load_history(dir_path: str | Path, history_days: int | None = None):
        now = pd.Timestamp.now("UTC")
        rows = []
        for p in Path(dir_path).rglob("*.fit"):
            s = parse_activity_fit(p)
            if (
                s
                and s.start_time is not None
                and (
                    history_days is None
                    or s.start_time >= now - pd.Timedelta(days=history_days)
                )
            ):
                rows.append(asdict(s))
        activities = pd.DataFrame(rows).sort_values("start_time").reset_index(drop=True)
        runs = activities[activities["is_run"]].copy().reset_index(drop=True)
        weekly_runs = runs.groupby("week", as_index=False).agg(
            runs=("file", "count"),
            distance_km=("distance_m", lambda x: x.sum() / 1000.0),
            duration_h=("duration_s", lambda x: x.sum() / 3600.0),
            elevation_m=("gain_m", "sum"),
            trimp=("trimp", "sum"),
            avg_hr=("avg_hr", "mean"),
            z1=("z1", "mean"),
            z2=("z2", "mean"),
            z3=("z3", "mean"),
            z4=("z4", "mean"),
            z5=("z5", "mean"),
        )
        return activities, runs, weekly_runs


    # 3) fatigue from all activities, planning from runs
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
        return daily


    def plot_load(daily: pd.DataFrame):
        import matplotlib.pyplot as plt

        ax = daily[["trimp", "ctl", "atl", "tsb"]].plot(
            figsize=(10, 5), secondary_y=["tsb"]
        )
        ax.set_ylabel("TRIMP / load")
        ax.right_ax.set_ylabel("TSB")
        plt.tight_layout()
        plt.show()


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


    def recent_pace_km_per_min(runs: pd.DataFrame, lookback: int = 10) -> float:
        x = runs.sort_values("start_time").tail(lookback).copy()
        x = x[(x["distance_m"] > 0) & (x["duration_s"] > 0)]
        if x.empty:
            return 0.10  # 6:00 min/km fallback
        return float((x["distance_m"].sum() / 1000.0) / (x["duration_s"].sum() / 60.0))


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


    def allocate_sessions(
        target_km: float,
        runs: int,
        pace_km_per_min: float,
        min_session_min: float = 20.0,
        min_session_km: float = 3.2,
        min_long_km: float = 6.0,
        min_quality_km: float = 4.0,
    ):
        min_by_time_km = pace_km_per_min * min_session_min
        session_min_km = max(min_session_km, min_by_time_km)

        runs = max(1, int(runs))
        max_feasible_runs = max(1, int(target_km // session_min_km))
        runs = min(runs, max_feasible_runs)

        if runs == 1:
            return {
                "type": "easy_only",
                "runs": 1,
                "sessions": [{"kind": "easy", "km": float(max(target_km, session_min_km))}],
            }

        can_long = target_km >= (runs - 1) * session_min_km + min_long_km
        can_quality = target_km >= (runs - 1) * session_min_km + min_quality_km

        sessions = []

        if (
            runs >= 3
            and can_long
            and can_quality
            and target_km >= min_long_km + min_quality_km + (runs - 2) * session_min_km
        ):
            long_km = max(min_long_km, target_km * 0.35)
            quality_km = max(min_quality_km, target_km * 0.20)
            remaining = target_km - long_km - quality_km
            easy_n = runs - 2
            easy_km = [remaining / easy_n] * easy_n
            if min(easy_km) < session_min_km:
                deficit = session_min_km * easy_n - remaining
                reducible = (long_km - min_long_km) + (quality_km - min_quality_km)
                if deficit <= reducible:
                    take_long = min(deficit, long_km - min_long_km)
                    long_km -= take_long
                    deficit -= take_long
                    take_quality = min(deficit, quality_km - min_quality_km)
                    quality_km -= take_quality
                    remaining = target_km - long_km - quality_km
                    easy_km = [remaining / easy_n] * easy_n
                else:
                    return {
                        "type": "easy_only",
                        "runs": runs,
                        "sessions": [
                            {"kind": "easy", "km": float(target_km / runs)}
                            for _ in range(runs)
                        ],
                    }
            sessions.append({"kind": "quality", "km": float(quality_km)})
            sessions += [{"kind": "easy", "km": float(x)} for x in easy_km]
            sessions.append({"kind": "long", "km": float(long_km)})
        else:
            base = target_km / runs
            if base < session_min_km:
                runs = max(1, int(target_km // session_min_km))
                runs = max(1, runs)
                base = target_km / runs
            sessions = [{"kind": "easy", "km": float(base)} for _ in range(runs)]

        return {
            "type": "mixed" if any(s["kind"] != "easy" for s in sessions) else "easy_only",
            "runs": len(sessions),
            "sessions": sessions,
        }


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


    HR_ZONES = {
        "Z1": (0.50, 0.60),
        "Z2": (0.60, 0.70),
        "Z3": (0.70, 0.80),
        "Z4": (0.80, 0.90),
        "Z5": (0.90, 1.00),
    }


    def hr_target(zone: str):
        lo, hi = HR_ZONES[zone]
        return {"target_type": "heart_rate", "zone": zone, "hrr_low": lo, "hrr_high": hi}


    def build_easy_session(target_km: float, pace_min_per_km: float):
        return {
            "kind": "easy",
            "steps": [
                {"type": "warmup", "min": 2, "target_type": "heart_rate", "zone": "Z1"},
                {
                    "type": "easy",
                    "km": max(16 / pace_min_per_km, target_km),
                    "target_type": "heart_rate",
                    "zone": "Z2",
                },
                {"type": "cooldown", "min": 2, "target_type": "heart_rate", "zone": "Z1"},
            ],
        }


    def build_long_session(target_km: float, pace_min_per_km: float):
        run_km = max(5.0, target_km - 4.0 / pace_min_per_km)  # subtract 2' WU + 2' CD
        return {
            "kind": "long",
            "steps": [
                {"type": "warmup", "min": 2, "target_type": "heart_rate", "zone": "Z1"},
                {"type": "easy", "km": run_km, "target_type": "heart_rate", "zone": "Z2"},
                {"type": "cooldown", "min": 2, "target_type": "heart_rate", "zone": "Z1"},
            ],
        }


    def build_quality_session(
        target_km: float, pace_min_per_km: float, state: str = "normal"
    ):

        prep_km = 8.0 / pace_min_per_km
        post_km = 8.0 / pace_min_per_km

        if state == "fatigued":
            repeats = 2
            on_min = 5
            off_min = 3
            zone = "Z3"
        elif target_km < 6.5:
            repeats = 6
            on_min = 2
            off_min = 2
            zone = "Z4"
        else:
            repeats = 3
            on_min = 6
            off_min = 3
            zone = "Z3"

        return {
            "kind": "quality",
            "steps": [
                {"type": "warmup", "min": 2, "target_type": "heart_rate", "zone": "Z1"},
                {"type": "easy", "km": prep_km, "target_type": "heart_rate", "zone": "Z2"},
                {
                    "type": "repeat",
                    "repeats": repeats,
                    "steps": [
                        {
                            "type": "interval",
                            "min": on_min,
                            "target_type": "heart_rate",
                            "zone": zone,
                        },
                        {
                            "type": "recovery",
                            "min": off_min,
                            "target_type": "heart_rate",
                            "zone": "Z1",
                        },
                    ],
                },
                {"type": "easy", "km": post_km, "target_type": "heart_rate", "zone": "Z2"},
                {"type": "cooldown", "min": 2, "target_type": "heart_rate", "zone": "Z1"},
            ],
        }


    def schedule_week(
        plan: dict,
        pace_min_per_km: float,
        state: str = "normal",
        preferred_days: list[str] | None = None,
    ):
        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        preferred = preferred_days[:] if preferred_days else days[:]
        out = {d: None for d in days}
        sessions = [
            dict(s)
            for s in sorted(
                plan["sessions"],
                key=lambda x: {"quality": 0, "long": 1, "easy": 2}[x["kind"]],
            )
        ]

        def gap_ok(day: str, hard_days: list[str]) -> bool:
            i = days.index(day)
            return all(abs(i - days.index(h)) >= 2 for h in hard_days)

        def place(kind: str, builder, hard_days: list[str]):
            for d in preferred:
                s = next(
                    (x for x in sessions if x["kind"] == kind and not x.get("used")), None
                )
                if s is None:
                    return
                if out[d] is None and (kind == "easy" or gap_ok(d, hard_days)):
                    s["used"] = True
                    out[d] = builder(s["km"])
                    if kind != "easy":
                        hard_days.append(d)
                    return

        hard_days = []
        place("quality", lambda km: build_quality_session(km, state), hard_days)
        place("long", lambda km: build_long_session(km, pace_min_per_km), hard_days)
        for _ in range(sum(s["kind"] == "easy" for s in sessions)):
            place("easy", lambda km: build_easy_session(km, pace_min_per_km), hard_days)
        return out


    def print_week(week: dict):
        for day in ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]:
            w = week.get(day)
            if w is None:
                print(f"{day}: REST")
                continue
            print(f"{day}: {w['kind'].upper()}")
            total_min = 0
            for step in w["steps"]:
                if step["type"] == "repeat":
                    print(f"  REPEAT {step['repeats']}x:")
                    for sub in step["steps"]:
                        unit = (
                            f"{sub['min']} min" if "min" in sub else f"{sub['km']:.2f} km"
                        )
                        print(f"    {sub['type']:10s} {unit:>8}  {sub['zone']}")
                        if "min" in sub:
                            total_min += sub["min"] * step["repeats"]
                else:
                    unit = f"{step['min']} min" if "min" in step else f"{step['km']:.2f} km"
                    print(f"  {step['type']:10s} {unit:>8}  {step['zone']}")
                    if "min" in step:
                        total_min += step["min"]
            print(f"  TOTAL: {total_min} min\n")


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


    def export_easy(workout, filename):

        builder = FitFileBuilder(auto_define=True)

        w = WorkoutMessage()
        w.workout_name = "Python Easy Run"
        builder.add(w)

        for step in workout["steps"]:
            s = WorkoutStepMessage()
            s.duration_type = WorkoutStepDuration.TIME
            s.target_type = WorkoutStepTarget.HEART_RATE
            s.intensity = Intensity.ACTIVE
            builder.add(s)

        builder.build().to_file(filename)


    def export_workout_from_template_like_structure(
        workout: dict,
        filename: str,
        workout_name: str = "Custom Run",
        serial_number: int = 1234567890,
        software_version: int = 2605,
    ):
        from fit_tool.fit_file_builder import FitFileBuilder
        from fit_tool.profile import profile_type as pt
        from fit_tool.profile.messages.file_creator_message import FileCreatorMessage
        from fit_tool.profile.messages.file_id_message import FileIdMessage
        from fit_tool.profile.messages.workout_message import WorkoutMessage
        from fit_tool.profile.messages.workout_step_message import WorkoutStepMessage

        builder = FitFileBuilder(auto_define=True)

        file_id = FileIdMessage()
        file_id.type = pt.FileType.WORKOUT
        file_id.manufacturer = pt.Manufacturer.GARMIN
        file_id.garmin_product = pt.GarminProduct.CONNECT
        file_id.serial_number = serial_number
        builder.add(file_id)

        file_creator = FileCreatorMessage()
        file_creator.hardware_version = 0
        file_creator.software_version = software_version
        builder.add(file_creator)

        intensity_map = {
            "warmup": pt.Intensity.WARMUP,
            "cooldown": pt.Intensity.COOLDOWN,
            "easy": pt.Intensity.ACTIVE,
            "steady": pt.Intensity.ACTIVE,
            "tempo": pt.Intensity.ACTIVE,
            "interval": pt.Intensity.ACTIVE,
            "recovery": pt.Intensity.RECOVERY,
        }
        hr_zone_map = {"Z1": 1, "Z2": 2, "Z3": 3, "Z4": 4, "Z5": 5}

        def set_duration(msg, step):
            if "min" in step and step["min"] is not None:
                msg.duration_type = pt.WorkoutStepDuration.TIME
                msg.duration_time = float(
                    step["min"] * 60.0 * 1000.0
                )  # keep your working scale
            elif "km" in step and step["km"] is not None:
                msg.duration_type = pt.WorkoutStepDuration.DISTANCE
                msg.duration_distance = float(step["km"] * 100.0)  # keep your working scale
            elif step.get("duration_type") == "hr_less_than":
                msg.duration_type = pt.WorkoutStepDuration.HR_LESS_THAN
                msg.duration_hr = int(step["duration_hr"])
            else:
                msg.duration_type = pt.WorkoutStepDuration.OPEN

        def set_target(msg, step):
            if step.get("target_type") == "heart_rate":
                msg.target_type = pt.WorkoutStepTarget.HEART_RATE
                msg.target_hr_zone = hr_zone_map[step["zone"]]
            else:
                msg.target_type = pt.WorkoutStepTarget.OPEN
                msg.target_value = 0

        # flatten steps Garmin-style:
        # children first, repeat controller after children
        flat_steps = []
        msg_idx = 0

        for step in workout["steps"]:
            if step["type"] != "repeat":
                s = WorkoutStepMessage()
                s.message_index = msg_idx
                s.intensity = intensity_map.get(step["type"], pt.Intensity.ACTIVE)
                set_duration(s, step)
                set_target(s, step)
                flat_steps.append(s)
                msg_idx += 1
                continue

            repeat_start_idx = msg_idx
            for sub in step["steps"]:
                t = WorkoutStepMessage()
                t.message_index = msg_idx
                t.intensity = intensity_map.get(sub["type"], pt.Intensity.ACTIVE)
                set_duration(t, sub)
                set_target(t, sub)
                flat_steps.append(t)
                msg_idx += 1

            r = WorkoutStepMessage()
            r.message_index = msg_idx
            r.duration_type = pt.WorkoutStepDuration.REPEAT_UNTIL_STEPS_CMPLT
            r.duration_step = repeat_start_idx
            r.target_type = pt.WorkoutStepTarget.OPEN
            r.target_repeat_steps = int(step["repeats"])
            print(
                "target_value =",
                next(f.get_value(0) for f in r.fields if f.name == "target_value"),
            )
            flat_steps.append(r)
            msg_idx += 1

        w = WorkoutMessage()
        w.workout_name = workout_name
        w.sport = pt.Sport.RUNNING
        w.sub_sport = pt.SubSport.GENERIC
        w.capabilities = pt.WorkoutCapabilities.TCX
        w.num_valid_steps = len(flat_steps)
        w.message_index = 0
        builder.add(w)

        for s in flat_steps:
            builder.add(s)

        builder.build().to_file(filename)


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
