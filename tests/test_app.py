"""Headless smoke test of the Streamlit app on every bundled sample (python -m pytest tests/)."""
from pathlib import Path
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def run_sample(index: int):
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    if index:
        at.sidebar.radio[0].set_value(at.sidebar.radio[0].options[index]).run()
    assert not at.exception, at.exception
    assert not at.error, [e.value for e in at.error]
    return at


def test_samples():
    for i in range(4):
        at = run_sample(i)
        assert len(at.tabs) == 6
        assert any("avoidable" in m.value for m in at.markdown)
