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


# ── 서귀포 휴게음식점이 Cloud 에서 빠지던 문제(2026-10-30): 받기 실패가 조용히 '파일 없음'이 되고, 한 번 실패하면 굳던 것 ──

CAFE_CSV = "업종명,업소명,소재지(도로명),소재지(지번),데이터기준일자\n휴게음식점,바다커피,제주특별자치도 서귀포시 성산읍 1,,2023-03-31\n".encode("utf-8-sig")


class _Resp:
    def __init__(self, raw):
        self.raw = raw

    def read(self):
        return self.raw

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _fake_opener(routes: dict, log: list):
    """routes: {url 조각: bytes | 예외}. 조각이 URL 에 들어 있는 첫 항목을 쓴다."""
    class Op:
        def open(self, req, timeout):
            url = req.full_url
            log.append(url)
            for frag, v in routes.items():
                if frag in url:
                    if isinstance(v, BaseException):
                        raise v
                    return _Resp(v)
            raise AssertionError("예상 못 한 주소 " + url)
    return Op()


def _patch_net(monkeypatch, mod, routes, log):
    monkeypatch.setattr(mod.urllib.request, "build_opener", lambda *a, **k: _fake_opener(routes, log))
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)


def test_cafe_fetch_direct_download_needs_one_request(tmp_path, monkeypatch):
    mod = fetch_public._load_script()
    log = []
    _patch_net(monkeypatch, mod, {"/api/file?id=4989&fileId=4079": CAFE_CSV}, log)
    assert "bytes" in mod.fetch_cafe(tmp_path)
    assert (tmp_path / mod.CAFE_NAME).read_bytes() == CAFE_CSV and len(log) == 1      # 272KB 상세 조회 없이 받는다


def test_cafe_fetch_retries_timeouts_then_succeeds(tmp_path, monkeypatch):
    mod = fetch_public._load_script()
    n = {"c": 0}

    class Op:
        def open(self, req, timeout):
            n["c"] += 1
            if n["c"] < 3:
                raise TimeoutError("timed out")
            return _Resp(CAFE_CSV)
    monkeypatch.setattr(mod.urllib.request, "build_opener", lambda *a, **k: Op())
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)
    mod.fetch_cafe(tmp_path)
    assert n["c"] == 3 and (tmp_path / mod.CAFE_NAME).exists()


def test_cafe_fetch_falls_back_to_lookup_when_ids_change(tmp_path, monkeypatch):
    import json
    mod = fetch_public._load_script()
    log = []
    meta = json.dumps({"file": {"fileInfos": [{"id": 7, "fileId": 8, "fileName": mod.CAFE_NAME}]}}).encode()
    _patch_net(monkeypatch, mod, {"id=4989": b"<html>not found</html>", "/api/data/view": meta, "id=7&fileId=8": CAFE_CSV}, log)
    mod.fetch_cafe(tmp_path)
    assert (tmp_path / mod.CAFE_NAME).read_bytes() == CAFE_CSV


def test_cafe_fetch_never_saves_a_non_csv_page(tmp_path, monkeypatch):
    mod = fetch_public._load_script()
    html = b"<html>blocked</html>"
    _patch_net(monkeypatch, mod, {"/api/file": html, "/api/data/view": OSError("blocked")}, [])
    try:
        mod.fetch_cafe(tmp_path)
        raise AssertionError("실패해야 합니다")
    except RuntimeError as e:
        assert "직접 받기" in str(e) and "상세 조회" in str(e)           # 두 경로의 사유가 모두 남는다
    assert not (tmp_path / mod.CAFE_NAME).exists()


def test_failure_reason_is_kept_for_the_ui(tmp_path):
    fetch_public.fetch_missing(tmp_path, script=_fake_script({"cafe": False, "bc": True}, [], fail={"cafe"}))
    assert "네트워크 끊김" in fetch_public.failure_reason("cafe") and fetch_public.failure_reason("bc") is None
    fetch_public.STATUS.clear()


def _store_without_public(syn_dir, tmp_path):
    from jangsa.store import DataStore
    s = DataStore()
    s.load_synthetic(syn_dir)
    s.load_public(tmp_path)
    return s


def test_skipped_real_step_has_no_real_data_label_and_shows_fetch_failure(syn_dir, tmp_path):
    """받기가 실패한 단계는 '[실데이터]' 표시 없이, 사유와 함께 건너뛰었다고 말한다."""
    from jangsa.agents.planner import build_plan
    from jangsa.llm import make_provider
    from jangsa.runlog import RunLog
    s = _store_without_public(syn_dir, tmp_path)
    fetch_public.STATUS.clear()
    fetch_public.STATUS.update({"cafe": "실패: URLError: timed out", "bc": "있음"})
    try:
        plan = build_plan("성산읍에 카페를 내면 어때?", s, make_provider("offline"), RunLog())
    finally:
        fetch_public.STATUS.clear()
    cafe_notes = [n for n in plan.notes if "휴게음식점" in n]
    assert cafe_notes and all("[실데이터]" not in n for n in cafe_notes)
    assert any("자동으로 받지 못했습니다" in n and "timed out" in n for n in cafe_notes)
