import recovercalc.cli as cli


def test_main_uses_default_history_days(monkeypatch):
    called = {}
    monkeypatch.setattr(
        cli.argparse.ArgumentParser,
        "parse_args",
        lambda self: type("A", (), {"history_days": 365})(),
    )
    monkeypatch.setattr(
        cli,
        "run_today",
        lambda history_days: called.setdefault("history_days", history_days),
    )
    cli.main()
    assert called["history_days"] == 365


def test_main_passes_custom_history_days(monkeypatch):
    called = {}
    monkeypatch.setattr(
        cli.argparse.ArgumentParser,
        "parse_args",
        lambda self: type("A", (), {"history_days": 30})(),
    )
    monkeypatch.setattr(
        cli,
        "run_today",
        lambda history_days: called.setdefault("history_days", history_days),
    )
    cli.main()
    assert called["history_days"] == 30
