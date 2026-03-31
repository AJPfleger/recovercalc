from .config import (
    MIN_SESSION_MIN,
    MIN_SESSION_KM,
    MIN_LONG_KM,
    MIN_QUALITY_KM,
    HR_ZONES,
)


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
