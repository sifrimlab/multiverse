import json
from unittest.mock import MagicMock, patch

import pytest

from multiverse.runner import docker_runner
from multiverse.runner.cli import emit_event


def test_emit_event_writes_jsonl_to_stderr(capsys):
    emit_event("job_start", run_id=42, job="ds_pca")
    payload = json.loads(capsys.readouterr().err)
    assert payload == {"event": "job_start", "job": "ds_pca", "run_id": 42}


def test_daemon_offline_emits_classified_error(capsys):
    with patch("docker.from_env", side_effect=RuntimeError("connection refused")):
        with pytest.raises(RuntimeError):
            docker_runner.get_docker_client()
    payload = json.loads(capsys.readouterr().err)
    assert payload["event"] == "error"
    assert payload["kind"] == "daemon_offline"


def test_image_pull_failure_emits_classified_error(capsys):
    client = MagicMock()
    client.images.get.side_effect = Exception("missing locally")
    client.images.pull.side_effect = RuntimeError("pull failed")
    with patch("docker.from_env", return_value=client), patch("multiverse.runner.docker_runner.get_db_connection") as mock_conn:
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None
        mock_conn.return_value.cursor.return_value = mock_cursor
        with pytest.raises(RuntimeError):
            docker_runner.ensure_image_prepared("missing:latest")
    events = [json.loads(line) for line in capsys.readouterr().err.splitlines() if line.strip()]
    assert any(e["kind"] == "image_pull_failed" for e in events)
