from pathlib import Path
import pandas as pd
import fitdecode
from dataclasses import asdict

from .metrics import _hr_zone_frac, _trimp_from_samples
from .models import ActivitySummary


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
