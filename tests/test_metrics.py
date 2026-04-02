import pandas as pd
from recovercalc.metrics import recent_pace_km_per_min, recent_pace_min_per_km


def test_recent_pace_nominal():
    runs = pd.DataFrame(
        {
            "start_time": [1, 2, 3],
            "distance_m": [1000, 2000, 3000],
            "duration_s": [300, 600, 900],  # all 5:00 min/km
        }
    )
    assert recent_pace_km_per_min(runs, 10) == 0.2
    assert recent_pace_min_per_km(runs, 10) == 5.0


def test_recent_pace_filters_invalid_rows():
    runs = pd.DataFrame(
        {
            "start_time": [1, 2, 3, 4],
            "distance_m": [1000, 0, 2000, 3000],
            "duration_s": [300, 100, 0, 900],
        }
    )
    # valid rows: (1000,300) and (3000,900) => 4 km / 20 min = 0.2 km/min
    assert recent_pace_km_per_min(runs, 10) == 0.2
    assert recent_pace_min_per_km(runs, 10) == 5.0


def test_recent_pace_fallback_when_no_valid_rows():
    runs = pd.DataFrame(
        {
            "start_time": [1, 2],
            "distance_m": [0, -5],
            "duration_s": [0, 100],
        }
    )
    assert recent_pace_km_per_min(runs, 10) == 0.10
    assert recent_pace_min_per_km(runs, 10) == 10.0


def test_recent_pace_respects_lookback_after_sorting():
    runs = pd.DataFrame(
        {
            "start_time": [30, 10, 20],
            "distance_m": [3000, 1000, 2000],
            "duration_s": [900, 300, 600],
        }
    )
    assert recent_pace_km_per_min(runs, lookback=2) == 0.2
