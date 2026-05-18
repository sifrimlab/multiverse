from __future__ import annotations

from streamlit.testing.v1 import AppTest


APP_PATH = "multiverse/gui.py"


def test_app_renders_query_param_navigation(monkeypatch):
    monkeypatch.setenv("MULTIVERSE_GUI_TELEMETRY", "0")
    at = AppTest.from_file(APP_PATH).run(timeout=10)

    assert not at.exception
    assert at.radio
    assert "Registry" in at.radio[0].options
    assert "Results" in at.radio[0].options


def test_top_nav_switches_to_results(monkeypatch):
    monkeypatch.setenv("MULTIVERSE_GUI_TELEMETRY", "0")
    at = AppTest.from_file(APP_PATH).run(timeout=10)

    at.radio[0].set_value("Results")
    at = at.run(timeout=10)

    assert not at.exception
    assert any(header.value == "Results" for header in at.header)
