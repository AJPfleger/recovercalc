from .config import MIN_SESSION_MIN, MIN_SESSION_KM, MIN_LONG_KM, MIN_QUALITY_KM

HR_ZONES = {
    "Z1": (0.50, 0.60),
    "Z2": (0.60, 0.70),
    "Z3": (0.70, 0.80),
    "Z4": (0.80, 0.90),
    "Z5": (0.90, 1.00),
}


def hr_target(zone: str):
    lo, hi = HR_ZONES[zone]
    return {
        "target_type": "heart_rate",
        "zone": zone,
        "hrr_low": lo,
        "hrr_high": hi,
    }


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
            {
                "type": "cooldown",
                "min": 2,
                "target_type": "heart_rate",
                "zone": "Z1",
            },
        ],
    }


def build_long_session(target_km: float, pace_min_per_km: float):
    run_km = max(5.0, target_km - 4.0 / pace_min_per_km)  # subtract 2' WU + 2' CD
    return {
        "kind": "long",
        "steps": [
            {"type": "warmup", "min": 2, "target_type": "heart_rate", "zone": "Z1"},
            {
                "type": "easy",
                "km": run_km,
                "target_type": "heart_rate",
                "zone": "Z2",
            },
            {
                "type": "cooldown",
                "min": 2,
                "target_type": "heart_rate",
                "zone": "Z1",
            },
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
            {
                "type": "easy",
                "km": prep_km,
                "target_type": "heart_rate",
                "zone": "Z2",
            },
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
            {
                "type": "easy",
                "km": post_km,
                "target_type": "heart_rate",
                "zone": "Z2",
            },
            {
                "type": "cooldown",
                "min": 2,
                "target_type": "heart_rate",
                "zone": "Z1",
            },
        ],
    }


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
        "type": (
            "mixed" if any(s["kind"] != "easy" for s in sessions) else "easy_only"
        ),
        "runs": len(sessions),
        "sessions": sessions,
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
                (x for x in sessions if x["kind"] == kind and not x.get("used")),
                None,
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
