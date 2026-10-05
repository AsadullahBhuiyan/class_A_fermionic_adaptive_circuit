"""Progress-display regression tests for standalone production runners."""

from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
import sys
import threading
import time

import pytest


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "00_WORKSPACE/CURRENT/final_production_new_designs"
RUNNERS = (
    ("04_maxmix_operator_cft", "g4_runner.py", "standalone_g4_progress_test"),
    ("05_pure_tangent_stability", "g5_runner.py", "standalone_g5_progress_test"),
    ("07_log_gram_alpha_scan", "log_gram_runner.py", "standalone_log_gram_progress_test"),
)


def _load_runner(bundle: str, filename: str, module_name: str):
    source_root = CAMPAIGN / bundle / "src"
    spec = importlib.util.spec_from_file_location(module_name, source_root / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    sys.path.insert(0, str(source_root))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(source_root))
    return module


@pytest.fixture(params=RUNNERS, ids=lambda item: item[0])
def runner(request):
    return _load_runner(*request.param)


class RecordingTqdm:
    instances = []
    messages = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.n = 0
        self.closed = False
        self.postfixes = []
        type(self).instances.append(self)

    def set_postfix_str(self, value, *, refresh=True):
        self.postfixes.append((value, refresh))

    def update(self, amount=1):
        self.n += amount

    def close(self):
        self.closed = True

    @classmethod
    def write(cls, message, *, file=None, **_):
        cls.messages.append(message)
        print(message, file=file)


def _patch_campaign(monkeypatch, runner, run_shard):
    cases = [{"case_id": "case-a"}, {"case_id": "case-b"}]
    monkeypatch.setattr(runner, "load_config", lambda _: {})
    if hasattr(runner, "expand_cases"):
        monkeypatch.setattr(
            runner, "expand_cases", lambda _config, *, pilot=False: list(cases)
        )
    else:
        monkeypatch.setattr(
            runner, "cases", lambda _config, pilot=False: list(cases)
        )
    monkeypatch.setattr(runner, "run_shard", run_shard)
    RecordingTqdm.instances = []
    RecordingTqdm.messages = []
    monkeypatch.setattr(runner, "tqdm", RecordingTqdm)


def test_parent_bar_counts_completed_and_checksum_skipped_filtered_jobs(
    runner, monkeypatch, capsys
):
    calls = []

    def run_shard(*args, **kwargs):
        calls.append((args, kwargs))
        status = "completed" if len(calls) % 2 else "verified_existing"
        return {"status": status, "data_path": f"shard-{len(calls)}.npz"}

    _patch_campaign(monkeypatch, runner, run_shard)
    assert runner.main(["production", "--case-id", "case-b", "--allow-cpu"]) == 0

    assert len(calls) == 5
    assert len(RecordingTqdm.instances) == 1
    bar = RecordingTqdm.instances[0]
    assert bar.kwargs["total"] == 5
    assert bar.kwargs["unit"] == "shard"
    assert bar.kwargs["file"] is sys.stderr
    assert bar.n == 5
    assert bar.closed
    assert all("case=case-b" in value for value, _ in bar.postfixes)

    stdout = capsys.readouterr().out
    payloads = [json.loads(line) for line in stdout.splitlines()]
    assert [payload["status"] for payload in payloads] == [
        "completed",
        "verified_existing",
        "completed",
        "verified_existing",
        "completed",
    ]


@pytest.mark.parametrize("failure", [RuntimeError("boom"), KeyboardInterrupt()])
def test_parent_bar_closes_without_advancing_failed_or_interrupted_jobs(
    runner, monkeypatch, failure
):
    calls = []

    def run_shard(*args, **kwargs):
        calls.append((args, kwargs))
        if len(calls) == 1:
            return {"status": "completed", "data_path": "first.npz"}
        raise failure

    _patch_campaign(monkeypatch, runner, run_shard)
    with pytest.raises(type(failure)):
        runner.main(["production", "--case-id", "case-a", "--allow-cpu"])

    bar = RecordingTqdm.instances[0]
    assert bar.n == 1
    assert bar.closed
    assert not any(
        thread.name == f"{runner.BUNDLE}-heartbeat" and thread.is_alive()
        for thread in threading.enumerate()
    )


def test_heartbeat_is_periodic_stderr_text_and_stops_cleanly(
    runner, monkeypatch
):
    monkeypatch.setattr(runner, "HEARTBEAT_SECONDS", 0.01)
    stream = io.StringIO()

    with runner._shard_heartbeat("case-live", 3, stream=stream):
        time.sleep(0.035)

    output = stream.getvalue()
    assert f"[{runner.BUNDLE}] heartbeat" in output
    assert "case=case-live" in output
    assert "shard=03" in output
    assert "elapsed=" in output
    assert not any(
        thread.name == f"{runner.BUNDLE}-heartbeat" and thread.is_alive()
        for thread in threading.enumerate()
    )


def test_engine_progress_is_owned_by_parent_bar(runner):
    source = Path(runner.__file__).read_text(encoding="utf-8")
    assert "progress=True" not in source
    assert "progress=False" in source or '"progress": False' in source

