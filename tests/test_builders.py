import recovercalc.builders
from recovercalc.builders import print_session


def test_print_training_formats_steps(capsys):
    """Verify a training session is printed in a readable terminal layout."""

    session = {
        "kind": "easy",
        "steps": [
            {"type": "warmup", "min": 2, "target_type": "heart_rate", "zone": "Z1"},
            {
                "type": "easy",
                "km": 1.4420129480457,
                "target_type": "heart_rate",
                "zone": "Z2",
            },
            {"type": "cooldown", "min": 2, "target_type": "heart_rate", "zone": "Z1"},
        ],
    }
    print_session(session)
    out = capsys.readouterr().out
    assert "Training: EASY" in out
    assert "1. WARMUP" in out and "2 min" in out and "heart_rate Z1" in out
    assert "2. EASY" in out and "1.44 km" in out and "heart_rate Z2" in out
    assert "3. COOLDOWN" in out and "2 min" in out and "heart_rate Z1" in out


def test_print_training_handles_missing_fields(capsys):
    """Verify missing optional step fields do not break terminal output."""

    session = {"kind": "unknown", "steps": [{"type": "easy"}]}
    print_session(session)
    out = capsys.readouterr().out
    assert "Training: UNKNOWN" in out
    assert "1. EASY" in out


def test_print_training_formats_repeat_blocks(capsys):
    """Verify nested repeat blocks are expanded and indented in the output."""

    session = {
        "kind": "quality",
        "steps": [
            {"type": "warmup", "min": 2, "target_type": "heart_rate", "zone": "Z1"},
            {
                "type": "repeat",
                "repeats": 2,
                "steps": [
                    {
                        "type": "interval",
                        "min": 2,
                        "target_type": "heart_rate",
                        "zone": "Z4",
                    },
                    {
                        "type": "recovery",
                        "min": 2,
                        "target_type": "heart_rate",
                        "zone": "Z1",
                    },
                ],
            },
        ],
    }

    print_session(session)
    out = capsys.readouterr().out

    assert "Training: QUALITY" in out
    assert " 1. WARMUP" in out
    assert " 2. REPEAT x2" in out
    assert "    1. INTERVAL" in out
    assert "    2. RECOVERY" in out
