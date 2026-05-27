from __future__ import annotations

from streamlit.testing.v1 import AppTest


APP_PATH = "multiverse/gui.py"


def _button_labels(at: AppTest) -> list[str]:
    return [getattr(button, "label", getattr(button, "value", "")) for button in at.button]


def test_app_renders_button_navigation(monkeypatch):
    monkeypatch.setenv("MULTIVERSE_GUI_TELEMETRY", "0")
    at = AppTest.from_file(APP_PATH).run(timeout=10)

    assert not at.exception
    labels = _button_labels(at)
    for label in ["Registry", "Configure", "Run", "Results", "Analysis"]:
        assert label in labels
    assert not at.radio


def test_top_nav_switches_to_results(monkeypatch):
    monkeypatch.setenv("MULTIVERSE_GUI_TELEMETRY", "0")
    at = AppTest.from_file(APP_PATH).run(timeout=10)

    labels = _button_labels(at)
    at.button[labels.index("Results")].click()
    at = at.run(timeout=10)

    assert not at.exception
    assert any(header.value == "Results" for header in at.header)


def test_legacy_execute_query_redirects_to_run(monkeypatch):
    monkeypatch.setenv("MULTIVERSE_GUI_TELEMETRY", "0")
    at = AppTest.from_file(APP_PATH)
    at.query_params["tab"] = "execute"
    at = at.run(timeout=10)

    assert not at.exception
    assert any(header.value == "Run" for header in at.header)
