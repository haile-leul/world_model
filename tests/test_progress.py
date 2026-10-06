import io
from world_model import progress


def test_log_progress_tracks_actual_work_and_early_exit(monkeypatch):
    stream = io.StringIO()
    monkeypatch.setattr(progress.sys, "stderr", stream)
    monkeypatch.delenv("WORLD_MODEL_PROGRESS", raising=False)
    with progress.Progress(10, "Inference", "step") as bar:
        bar.update(3, episode_return="1.25")
    output = stream.getvalue()
    assert "0/10" in output and "3/10" in output
    assert "ended early" in output and "10/10" not in output
    assert "episode_return=1.25" in output


def test_disabled_progress_is_silent(monkeypatch):
    stream = io.StringIO()
    monkeypatch.setattr(progress.sys, "stderr", stream)
    monkeypatch.setenv("WORLD_MODEL_PROGRESS", "0")
    progress.stage("hidden")
    assert list(progress.track([1, 2], "hidden")) == [1, 2]
    assert stream.getvalue() == ""


def test_error_does_not_report_success(monkeypatch):
    import pytest

    stream = io.StringIO()
    monkeypatch.setattr(progress.sys, "stderr", stream)
    monkeypatch.delenv("WORLD_MODEL_PROGRESS", raising=False)
    with pytest.raises(RuntimeError):
        with progress.Progress(2, "Training") as bar:
            bar.update()
            raise RuntimeError("failure")
    assert "interrupted" in stream.getvalue()
    assert "[complete]" not in stream.getvalue()
