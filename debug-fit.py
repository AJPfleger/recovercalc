import json
from pathlib import Path
import fitdecode

KEEP = {
    "file_id",
    "file_creator",
    "workout",
    "workout_step",
    "memo_glob",
    "exercise_title",
}


def dump_fit_template(fit_path: str, out_json: str | None = None):
    messages = []
    with fitdecode.FitReader(fit_path) as fit:
        for i, frame in enumerate(fit):
            if not isinstance(frame, fitdecode.FitDataMessage):
                continue
            if frame.name not in KEEP:
                continue
            fields = {}
            for f in frame.fields:
                v = f.value
                try:
                    json.dumps(v)
                except TypeError:
                    v = str(v)
                fields[f.name] = v
            messages.append({"idx": i, "name": frame.name, "fields": fields})
    if out_json:
        Path(out_json).write_text(json.dumps(messages, indent=2, default=str))
    return messages


# tmpl = dump_fit_template("Run_Workout_HR_test_workout.fit", "garmin_workout_template.json")
# tmpl = dump_fit_template("Easy_Run_workout_01.fit", "garmin_workout_template.json")
# tmpl = dump_fit_template("easy_template_based.fit", "garmin_workout_template.json")
tmpl = dump_fit_template("Run_Workout_workout_01.fit", "garmin_workout_template.json")
for m in tmpl:
    print(m["name"], m["fields"])


def summarize_workout_template(messages):
    out = []
    for m in messages:
        f = m["fields"]
        if m["name"] == "workout":
            out.append(
                (
                    "workout",
                    {
                        k: f.get(k)
                        for k in [
                            "workout_name",
                            "sport",
                            "num_valid_steps",
                            "capabilities",
                        ]
                    },
                )
            )
        elif m["name"] == "workout_step":
            out.append(
                (
                    "workout_step",
                    {
                        k: f.get(k)
                        for k in [
                            "message_index",
                            "wkt_step_name",
                            "duration_type",
                            "duration_time",
                            "duration_distance",
                            "duration_hr",
                            "target_type",
                            "target_value",
                            "target_hr_zone",
                            "custom_target_value_low",
                            "custom_target_value_high",
                            "intensity",
                            "notes",
                        ]
                    },
                )
            )
        else:
            out.append((m["name"], f))
    return out


summary = summarize_workout_template(tmpl)
for x in summary:
    print(x)


# first target: verify required structure before writing anything new
required = ["file_id", "file_creator", "workout", "workout_step"]
present = [m["name"] for m in tmpl]
print("required_ok =", all(x in present for x in required))
print("message_order =", present)
print("step_count =", sum(m["name"] == "workout_step" for m in tmpl))


from fit_tool.profile.messages.workout_step_message import WorkoutStepMessage

s = WorkoutStepMessage()
for f in s.fields:
    if "repeat" in f.name or "duration" in f.name:
        print("name=", f.name)
        print("attrs=", [a for a in dir(f) if not a.startswith("_")])
        print("base_type=", getattr(getattr(f, "base_type", None), "name", None))
        print("---")


from fit_tool.profile import profile_type as pt
from fit_tool.profile.messages.workout_step_message import WorkoutStepMessage

r = WorkoutStepMessage()
r.message_index = 3
r.duration_type = pt.WorkoutStepDuration.REPEAT_UNTIL_STEPS_CMPLT
r.duration_step = 1
r.repeat_steps = 6

for f in r.fields:
    if "repeat" in f.name or "duration" in f.name:
        try:
            v = f.get_value(0)
        except Exception as e:
            v = repr(e)
        print(f.name, v)


from fit_tool.profile import profile_type as pt
from fit_tool.profile.messages.workout_step_message import WorkoutStepMessage

r = WorkoutStepMessage()
r.message_index = 3
r.duration_type = pt.WorkoutStepDuration.REPEAT_UNTIL_STEPS_CMPLT
r.duration_step = 1

print("has repeat_steps:", hasattr(r, "repeat_steps"))
print("dir contains:", [x for x in dir(r) if "repeat" in x.lower()])

r.repeat_steps = 6
for f in r.fields:
    try:
        print(f.name, f.get_value(0))
    except Exception:
        pass
