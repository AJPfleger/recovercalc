from dataclasses import dataclass
import pandas as pd


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
