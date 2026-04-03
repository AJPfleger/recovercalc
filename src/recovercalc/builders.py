from .config import (
    MIN_SESSION_MIN,
    MIN_SESSION_KM,
    MIN_LONG_KM,
    MIN_QUALITY_KM,
    HR_ZONES,
    WARMUP_TIME,
    COOLDOWN_TIME,
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
    core_distance = (
        max(MIN_SESSION_MIN / pace_min_per_km, target_km)
        - (WARMUP_TIME + COOLDOWN_TIME) / pace_min_per_km
    )

    return {
        "kind": "easy",
        "steps": [
            {
                "type": "warmup",
                "min": WARMUP_TIME,
                "target_type": "heart_rate",
                "zone": "Z1",
            },
            {
                "type": "easy",
                "km": core_distance / 2,
                "target_type": "heart_rate",
                "zone": "Z2",
            },
            {
                "type": "easy",
                "km": core_distance / 2,
                "target_type": "heart_rate",
                "zone": "Z2",
            },
            {
                "type": "cooldown",
                "min": COOLDOWN_TIME,
                "target_type": "heart_rate",
                "zone": "Z1",
            },
        ],
    }


def build_long_session(target_km: float, pace_min_per_km: float):
    core_distance = (
        max(MIN_LONG_KM, target_km) - (WARMUP_TIME + COOLDOWN_TIME) / pace_min_per_km
    )

    return {
        "kind": "long",
        "steps": [
            {
                "type": "warmup",
                "min": WARMUP_TIME,
                "target_type": "heart_rate",
                "zone": "Z1",
            },
            {
                "type": "easy",
                "km": core_distance / 2,
                "target_type": "heart_rate",
                "zone": "Z2",
            },
            {
                "type": "easy",
                "km": core_distance / 2,
                "target_type": "heart_rate",
                "zone": "Z2",
            },
            {
                "type": "cooldown",
                "min": COOLDOWN_TIME,
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
            {
                "type": "warmup",
                "min": WARMUP_TIME,
                "target_type": "heart_rate",
                "zone": "Z1",
            },
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
                "min": COOLDOWN_TIME,
                "target_type": "heart_rate",
                "zone": "Z1",
            },
        ],
    }


def print_session(session: dict) -> None:
    """Print a formatted textual representation of a training session.

    Displays the session type and its ordered steps in a readable
    tabular layout. Each step shows the activity type, duration or
    distance, and the associated training target (for example heart
    rate zone). Intended for terminal output.
    """

    def _print_step(step: dict, idx: str = "", indent: int = 0) -> None:
        prefix = " " * indent
        step_type = step.get("type", "?").upper()
        target = step.get("target_type", "")
        zone = step.get("zone", "")

        if step.get("type") == "repeat":
            repeats = step.get("repeats", "?")
            print(f"{prefix}{idx}REPEAT x{repeats}")
            for j, substep in enumerate(step.get("steps", []), 1):
                _print_step(substep, idx=f"{j}. ", indent=indent + 4)
            return

        if "km" in step:
            value = f"{step['km']:.2f} km"
        elif "min" in step:
            value = f"{step['min']} min"
        else:
            value = ""

        print(f"{prefix}{idx}{step_type:<10} {value:<10} {target} {zone}".rstrip())

    kind = session.get("kind", "unknown").upper()
    steps = session.get("steps", [])

    print(f"\nTraining: {kind}")
    print("-" * 40)
    for i, step in enumerate(steps, 1):
        _print_step(step, idx=f"{i:2d}. ")
    print("-" * 40)
