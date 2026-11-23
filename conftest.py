"""Offline-only pytest: no production training, background jobs or live HTTP."""
import os
os.environ['CAPITEX_DISABLE_BACKGROUND_GRADER']='1'
import pytest

# This old manual script resets the real running prototype wallet, so it is not
# an automated test. Equivalent isolated transaction tests run in the suite.
collect_ignore=['test_trailing_stop.py']

def pytest_collection_modifyitems(items):
    for item in items:
        if 'LSTMIntegrationTests' in item.nodeid:
            item.add_marker(pytest.mark.skip(reason='Upgrade guardrail: no LSTM training; existing synthetic-training tests retained for explicit separate use.'))

@pytest.fixture(autouse=True)
def offline_http(monkeypatch):
    def denied(*args,**kwargs):
        raise RuntimeError('Live HTTP disabled in pytest; inject a fake provider.')
    import requests
    monkeypatch.setattr(requests.sessions.Session,'send',denied)
    import urllib.request
    monkeypatch.setattr(urllib.request,'urlopen',denied)
    try:
        import curl_cffi.requests
        monkeypatch.setattr(curl_cffi.requests.Session,'request',denied)
    except ImportError:
        pass
