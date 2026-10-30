"""Cloud 데모의 첫 실행 받기: 네트워크는 가짜로 바꿔 확인한다."""
from types import SimpleNamespace

from jangsa import fetch_public


def _fake_script(have: dict, fetch_log: list, fail: set = frozenset()):
    def mk(key):
        def fetch(out):
            fetch_log.append(key)
            if key in fail:
                raise RuntimeError("네트워크 끊김")
            return "ok"
        return (lambda out: have[key]), fetch, key
    return SimpleNamespace(JOBS={"cafe": mk("cafe"), "bc": mk("bc")}, TIMEOUT_CAP=None)


def test_fetches_only_missing_and_sets_timeout(tmp_path):
    log = []
    sc = _fake_script({"cafe": True, "bc": False}, log)
    res = fetch_public.fetch_missing(tmp_path, script=sc)
    assert res == {"cafe": "있음", "bc": "ok"} and log == ["bc"]
    assert sc.TIMEOUT_CAP == fetch_public.REQUEST_TIMEOUT == 20


def test_failure_does_not_raise(tmp_path):
    log = []
    res = fetch_public.fetch_missing(tmp_path, script=_fake_script({"cafe": False, "bc": False}, log, fail={"bc"}))
    assert res["cafe"] == "ok" and res["bc"].startswith("실패") and sorted(log) == ["bc", "cafe"]


def test_real_script_loads_with_both_jobs():
    mod = fetch_public._load_script()
    assert set(mod.JOBS) == {"cafe", "bc"} and mod.TIMEOUT_CAP is None


def test_timeout_cap_applies(monkeypatch):
    mod = fetch_public._load_script()
    seen = []

    class Op:
        def open(self, req, timeout):
            seen.append(timeout)
            raise OSError("x")
    mod.TIMEOUT_CAP = 20
    try:
        mod._open(Op(), "http://example.invalid", timeout=300)
    except OSError:
        pass
    assert seen == [20]
