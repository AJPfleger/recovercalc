import pandas as pd
import pytest
import recovercalc.decision as decision
from recovercalc.decision import (
    _local_day,
    _today_local,
    _classify_run,
    next_week_targets,
    _rest_reason,
    _easy_reason,
    decide_today,
)

TEST_TZ = "Pacific/Chatham"


@pytest.fixture(autouse=True)
def patch_tz(monkeypatch):
    monkeypatch.setattr(decision, "LOCAL_TZ", TEST_TZ)


def test_local_day_converts_utc_series_to_local_midnight():
    ts = pd.Series([pd.Timestamp("2024-01-01 23:30:00+00:00")])
    out = _local_day(ts)
    # adjust date if LOCAL_TZ differs
    expected = pd.Series([pd.Timestamp("2024-01-02 00:00:00", tz=TEST_TZ)])
    pd.testing.assert_series_equal(out.reset_index(drop=True), expected)


def test_today_local_none_returns_local_midnight():
    out = _today_local()
    assert out.tz is not None
    assert out.hour == 0 and out.minute == 0 and out.second == 0


def test_today_local_naive_timestamp_is_localized_and_floored():
    out = _today_local(pd.Timestamp("2024-03-05 12:34:56"))
    assert out == pd.Timestamp("2024-03-05 00:00:00", tz=TEST_TZ)


def test_today_local_aware_timestamp_is_converted_and_floored():
    out = _today_local(pd.Timestamp("2024-03-05 23:30:00+00:00"))
    # adjust date if LOCAL_TZ differs
    expected = pd.Timestamp("2024-03-06 00:00:00", tz=TEST_TZ)
    assert out == expected


def test_classify_run_returns_long_when_duration_and_trimp_are_high():
    row = pd.Series(
        {"duration_s": 120 * 60, "trimp": 999, "z3": 0.0, "z4": 0.0, "z5": 0.0}
    )
    assert _classify_run(row) == "LONG"


def test_classify_run_returns_quality_from_z4_z5_path():
    row = pd.Series(
        {"duration_s": 30 * 60, "trimp": 999, "z3": 0.0, "z4": 999, "z5": 999}
    )
    assert _classify_run(row) == "QUALITY"


def test_classify_run_returns_quality_from_z3_path():
    row = pd.Series(
        {"duration_s": 30 * 60, "trimp": 999, "z3": 999, "z4": 0.0, "z5": 0.0}
    )
    assert _classify_run(row) == "QUALITY"


def test_classify_run_returns_easy_when_thresholds_are_not_met():
    row = pd.Series(
        {"duration_s": 20 * 60, "trimp": 0.0, "z3": 0.0, "z4": 0.0, "z5": 0.0}
    )
    assert _classify_run(row) == "EASY"


def test_next_week_targets_reduces_load_when_fatigued():
    """Verify targets are reduced when TSB indicates high fatigue."""
    daily = pd.DataFrame({"ctl": [50.0], "tsb": [-15.0]})
    weekly = pd.DataFrame(
        {
            "week": [1, 2, 3, 4],
            "distance_km": [20, 22, 24, 26],
            "trimp": [200, 210, 220, 230],
            "runs": [3, 4, 4, 5],
        }
    )

    out = next_week_targets(daily, weekly)

    assert out["target_km"] < 23  # below recent mean due to reduction
    assert out["runs"] >= 3


def test_next_week_targets_increases_load_when_recovered():
    """Verify targets increase when TSB indicates good recovery."""
    daily = pd.DataFrame({"ctl": [50.0], "tsb": [10.0]})
    weekly = pd.DataFrame(
        {
            "week": [1, 2, 3, 4],
            "distance_km": [20, 20, 20, 20],
            "trimp": [200, 200, 200, 200],
            "runs": [3, 3, 3, 3],
        }
    )

    out = next_week_targets(daily, weekly)

    assert out["target_km"] > 20
    assert out["runs"] == 3


def test_next_week_targets_clips_runs_to_bounds():
    """Verify run count is clipped to configured limits."""
    daily = pd.DataFrame({"ctl": [50.0], "tsb": [0.0]})
    weekly = pd.DataFrame(
        {
            "week": [1, 2, 3, 4],
            "distance_km": [10, 10, 10, 10],
            "trimp": [100, 100, 100, 100],
            "runs": [10, 10, 10, 10],
        }
    )

    out = next_week_targets(daily, weekly, min_runs=3, max_runs=5)

    assert out["runs"] == 5


def test_next_week_targets_respects_max_weekly_km():
    """Verify weekly distance target is capped by max_weekly_km."""
    daily = pd.DataFrame({"ctl": [50.0], "tsb": [10.0]})
    weekly = pd.DataFrame(
        {
            "week": [1, 2, 3, 4],
            "distance_km": [30, 30, 30, 30],
            "trimp": [300, 300, 300, 300],
            "runs": [4, 4, 4, 4],
        }
    )

    out = next_week_targets(daily, weekly, max_weekly_km=25.0)

    assert out["target_km"] <= 25.0


@pytest.mark.parametrize(
    "daily_state, run_state, expected",
    [
        (
            {"yesterday_trimp_all": 120.0, "tsb_today": 0.0},
            {"runs_empty": False, "days_since_run": 5},
            True,
        ),
        (
            {"yesterday_trimp_all": 50.0, "tsb_today": -6.0},
            {"runs_empty": False, "days_since_run": 5},
            True,
        ),
        (
            {"yesterday_trimp_all": 50.0, "tsb_today": 0.0},
            {"runs_empty": False, "days_since_run": 1},
            True,
        ),
        (
            {"yesterday_trimp_all": 50.0, "tsb_today": 0.0},
            {"runs_empty": False, "days_since_run": 3},
            False,
        ),
        (
            {"yesterday_trimp_all": 50.0, "tsb_today": 0.0},
            {"runs_empty": True, "days_since_run": 0},
            False,
        ),
    ],
)
def test_rest_reason(monkeypatch, daily_state, run_state, expected):
    """Verify REST is triggered only by the defined hard safety rules."""
    monkeypatch.setattr(decision, "VERY_HIGH_LOAD_TRIMP", 100.0)
    monkeypatch.setattr(decision, "EASY_TSB_MIN", -5.0)
    monkeypatch.setattr(decision, "MIN_DAYS_BETWEEN_RUNS", 2)
    assert _rest_reason(daily_state, run_state) is expected


@pytest.mark.parametrize(
    "daily_state, run_state, expected",
    [
        (
            {"tsb_today": -10.0, "yesterday_trimp_all": 0.0},
            {
                "runs_empty": False,
                "days_since_run": 5,
                "days_since_long": 5,
                "days_since_quality": 5,
            },
            True,
        ),
        (
            {"tsb_today": 10.0, "yesterday_trimp_all": 0.0},
            {
                "runs_empty": True,
                "days_since_run": 0,
                "days_since_long": 999,
                "days_since_quality": 999,
            },
            True,
        ),
        (
            {"tsb_today": 10.0, "yesterday_trimp_all": 9999.0},
            {
                "runs_empty": False,
                "days_since_run": 5,
                "days_since_long": 5,
                "days_since_quality": 5,
            },
            False,
        ),  # REST case handled earlier
        (
            {"tsb_today": 10.0, "yesterday_trimp_all": 0.0},
            {
                "runs_empty": False,
                "days_since_run": 5,
                "days_since_long": 1,
                "days_since_quality": 5,
            },
            True,
        ),
        (
            {"tsb_today": 10.0, "yesterday_trimp_all": 0.0},
            {
                "runs_empty": False,
                "days_since_run": 5,
                "days_since_long": 5,
                "days_since_quality": 1,
            },
            True,
        ),
        (
            {"tsb_today": 10.0, "yesterday_trimp_all": 0.0},
            {
                "runs_empty": False,
                "days_since_run": 5,
                "days_since_long": 5,
                "days_since_quality": 5,
            },
            False,
        ),
    ],
)
def test_easy_reason(monkeypatch, daily_state, run_state, expected):
    """Verify EASY is selected only when recovery or hard-session spacing rules require it."""
    monkeypatch.setattr(decision, "QUALITY_TSB_MIN", 0.0)
    monkeypatch.setattr(decision, "LONG_TSB_MIN", 0.0)
    monkeypatch.setattr(decision, "HIGH_LOAD_TRIMP", 100.0)
    monkeypatch.setattr(decision, "MIN_DAYS_AFTER_LONG", 2)
    monkeypatch.setattr(decision, "MIN_DAYS_AFTER_QUALITY", 2)
    assert _easy_reason(daily_state, run_state) is expected


def test_decide_today_returns_rest_after_very_high_load_yesterday(monkeypatch):
    """Verify a very hard previous day forces REST immediately."""
    monkeypatch.setattr(decision, "VERY_HIGH_LOAD_TRIMP", 100.0)

    idx = pd.to_datetime(["2024-03-01 00:00:00+00:00", "2024-03-02 00:00:00+00:00"])
    daily = pd.DataFrame({"tsb": [0.0, 0.0], "trimp": [50.0, 150.0]}, index=idx)
    runs = pd.DataFrame({"start_time": [pd.Timestamp("2024-03-01 06:00:00+00:00")]})

    out = decide_today(daily, runs, today=pd.Timestamp("2024-03-03 12:00:00+00:00"))
    assert out == "REST"


def test_decide_today_returns_easy_when_no_runs_exist(monkeypatch):
    """Verify empty run history falls back to EASY once safety checks pass."""
    monkeypatch.setattr(decision, "VERY_HIGH_LOAD_TRIMP", 9999.0)
    monkeypatch.setattr(decision, "EASY_TSB_MIN", -999.0)
    monkeypatch.setattr(decision, "QUALITY_TSB_MIN", 999.0)
    monkeypatch.setattr(decision, "LONG_TSB_MIN", 999.0)

    idx = pd.to_datetime(["2024-03-01 00:00:00+00:00", "2024-03-02 00:00:00+00:00"])
    daily = pd.DataFrame({"tsb": [10.0, 10.0], "trimp": [10.0, 10.0]}, index=idx)
    runs = pd.DataFrame(columns=["start_time"])

    out = decide_today(daily, runs, today=pd.Timestamp("2024-03-03 12:00:00+00:00"))
    assert out == "EASY"


def test_decide_today_returns_quality_when_quality_conditions_are_met(monkeypatch):
    """Verify QUALITY is chosen when recovery and spacing satisfy quality rules."""
    monkeypatch.setattr(decision, "VERY_HIGH_LOAD_TRIMP", 9999.0)
    monkeypatch.setattr(decision, "HIGH_LOAD_TRIMP", 9999.0)
    monkeypatch.setattr(decision, "EASY_TSB_MIN", -999.0)
    monkeypatch.setattr(decision, "QUALITY_TSB_MIN", 0.0)
    monkeypatch.setattr(decision, "LONG_TSB_MIN", 0.0)
    monkeypatch.setattr(decision, "MIN_DAYS_BETWEEN_RUNS", 1)
    monkeypatch.setattr(decision, "MIN_DAYS_AFTER_QUALITY", 2)
    monkeypatch.setattr(decision, "MIN_DAYS_AFTER_LONG", 2)

    idx = pd.to_datetime(["2024-03-01 00:00:00+00:00", "2024-03-02 00:00:00+00:00"])
    daily = pd.DataFrame({"tsb": [10.0, 10.0], "trimp": [10.0, 10.0]}, index=idx)
    runs = pd.DataFrame(
        [
            {
                "start_time": pd.Timestamp("2024-02-27 06:00:00+00:00"),
                "duration_s": 1800,
                "trimp": 10.0,
                "z3": 0.0,
                "z4": 0.0,
                "z5": 0.0,
            },
            {
                "start_time": pd.Timestamp("2024-02-29 06:00:00+00:00"),
                "duration_s": 1800,
                "trimp": 10.0,
                "z3": 0.0,
                "z4": 0.0,
                "z5": 0.0,
            },
        ]
    )

    out = decide_today(
        daily,
        runs,
        today=pd.Timestamp("2024-03-03 12:00:00+00:00"),
        quality_gap_days=3,
        long_gap_days=99,
    )
    assert out == "QUALITY"


def test_decide_today_returns_long_when_long_conditions_are_met(monkeypatch):
    """Verify LONG is chosen when long-run spacing is due and quality spacing is respected."""
    monkeypatch.setattr(decision, "VERY_HIGH_LOAD_TRIMP", 9999.0)
    monkeypatch.setattr(decision, "HIGH_LOAD_TRIMP", 9999.0)
    monkeypatch.setattr(decision, "EASY_TSB_MIN", -999.0)
    monkeypatch.setattr(decision, "QUALITY_TSB_MIN", 999.0)
    monkeypatch.setattr(decision, "LONG_TSB_MIN", 0.0)
    monkeypatch.setattr(decision, "MIN_DAYS_BETWEEN_RUNS", 1)
    monkeypatch.setattr(decision, "MIN_DAYS_AFTER_QUALITY", 2)
    monkeypatch.setattr(decision, "MIN_DAYS_AFTER_LONG", 2)

    idx = pd.to_datetime(["2024-03-01 00:00:00+00:00", "2024-03-02 00:00:00+00:00"])
    daily = pd.DataFrame({"tsb": [10.0, 10.0], "trimp": [10.0, 10.0]}, index=idx)
    runs = pd.DataFrame(
        [
            {
                "start_time": pd.Timestamp("2024-02-27 06:00:00+00:00"),
                "duration_s": 1800,
                "trimp": 10.0,
                "z3": 0.0,
                "z4": 0.0,
                "z5": 0.0,
            },
            {
                "start_time": pd.Timestamp("2024-02-29 06:00:00+00:00"),
                "duration_s": 1800,
                "trimp": 10.0,
                "z3": 0.0,
                "z4": 0.0,
                "z5": 0.0,
            },
        ]
    )

    out = decide_today(
        daily,
        runs,
        today=pd.Timestamp("2024-03-03 12:00:00+00:00"),
        quality_gap_days=99,
        long_gap_days=3,
    )
    assert out == "LONG"
