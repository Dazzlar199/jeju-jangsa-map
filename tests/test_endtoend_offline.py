import pytest

from jangsa.agents.rules import parse_question
from jangsa.llm import make_provider
from jangsa.pipeline import analyze, display_numbers
from jangsa.report import build_report_html
from jangsa.config import ATTRIBUTION

QUESTIONS = [
    ("애월읍에 카페를 내면 어때?", "location", "애월읍", "카페"),
    ("노형동 한식집 매출이 동네 평균보다 어때?", "benchmark", "노형동", "한식"),
    ("성산읍에 카페가 너무 많은가요?", "competition", "성산읍", "카페"),
    ("중문동 카페는 몇 시에 제일 잘 돼?", "time_pattern", "중문동", "카페"),
    ("구좌읍 일식집은 비 오는 달에 매출이 줄어?", "season_weather", "구좌읍", "일식"),
    ("제주시에서 편의점 하기 좋은 동네는?", "location", None, "편의점"),
    ("한림읍 빵집 주말에 잘 돼요?", "time_pattern", "한림읍", "베이커리"),
]


@pytest.mark.parametrize("q,intent,emd,ind", QUESTIONS)
def test_offline_end_to_end(store, q, intent, emd, ind):
    p = parse_question(q)
    assert (p.intent, p.emd, p.industry) == (intent, emd, ind)
    r = analyze(q, store, make_provider("offline"))
    assert r.log.n_calls == 0 and r.plan.mode == "offline"
    ran = [s for s in r.steps if not s.skipped and s.step.source != "local"]
    assert len(ran) >= 2 and all(s.error == "" for s in ran)
    assert r.explanation.headline and r.explanation.points and r.explanation.limits
    assert r.explanation.source == "template"
    assert r.synthetic                                              # 가상 데이터 표시
    assert any("가상" in l for l in r.explanation.limits)
    for s in ran:
        assert s.gate is not None and s.gate.passed, (s.step.id, s.gate.summary)
        assert s.sql_run.strip().upper().startswith(("SELECT", "WITH")) and "LIMIT" in s.sql_run.upper()
    steps = {s.step.id: s for s in ran}
    # 엄격 모드: 원본 불가 표 결과는 반출용 표에 단순집계가 없다
    for s in ran:
        if s.step.source != "local" and s.gate.level == "restricted":
            assert not (set(s.df_export.columns) & set(s.gate.dropped_for_export))
    # 근거표 번호는 실제 통과한 표를 가리킨다
    nums = display_numbers(r)
    for pt in r.explanation.points:
        assert all(n in nums.values() for n in pt["tables"])


def test_real_public_steps_are_labelled_and_separate(store):
    r = analyze("애월읍에 카페를 내면 어때?", store, make_provider("offline"))
    real = [s for s in r.steps if s.step.id.startswith("real_")]
    assert real and all(s.origin == "public" for s in real)
    assert all(s.origin == "synthetic" for s in r.steps if not s.step.id.startswith("real_") and s.step.source != "local")
    assert all(s.step.title.startswith("[실데이터]") for s in real)
    r3 = analyze("애월읍에 카페를 내면 어때?", store, make_provider("offline"))
    assert not any(s.step.id == "real_cafe_density" for s in r3.steps)      # 제주시는 해당 자료가 없음


def test_resident_hourly_chart_has_values_for_aewol_cafe(store):
    """'애월읍 카페' 결과의 [실데이터] 도민 유동인구 그래프가 비어 나오던 결함(반올림으로 0) 회귀 테스트."""
    if not store.has("resident_hourly"):
        pytest.skip("도민 유동인구 파일 없음")
    r = analyze("애월읍에 카페를 내면 어때?", store, make_provider("offline"))
    s = next(x for x in r.steps if x.step.id == "real_resident_hourly")
    assert s.gate.passed and len(s.df) == 5
    assert (s.df["pop_share_pct"] > 0).all() and abs(s.df["pop_share_pct"].sum() - 100) < 1.0
    assert (s.df["pop_avg"] > 1).all()                                       # 사람 수 평균(소수 둘째 자리 반올림으로 0이 되지 않음)


def test_real_cafe_density_when_file_present(store):
    if not store.has("cafe_license"):
        pytest.skip("fetch_public_data.py 로 받는 파일 없음")
    r2 = analyze("성산읍에 카페가 너무 많은가요?", store, make_provider("offline"))
    assert any(s.step.id == "real_cafe_density" for s in r2.steps)          # 서귀포 카페 → 공개 휴게음식점 실데이터


def test_missing_public_files_are_skipped_with_guidance(syn_dir, tmp_path):
    """공개 실데이터 파일이 하나도 없어도 앱·분석이 돌고, 빠진 [실데이터] 단계는 '파일 없음 — 받는 법'으로 안내된다."""
    from jangsa.store import DataStore
    s = DataStore()
    s.load_synthetic(syn_dir)
    s.load_public(tmp_path)                                                  # 빈 폴더
    assert not any(s.has(t) for t in ("cafe_license", "bc_tour_time", "bc_tour_dow", "resident_hourly", "bus_stops"))
    r = analyze("성산읍에 카페가 너무 많은가요?", s, make_provider("offline"))
    ran = [x for x in r.steps if not x.skipped]
    assert len(ran) >= 2 and all(x.error == "" and x.gate.passed for x in ran)
    assert not any(x.step.id.startswith("real_") for x in r.steps)
    assert any("파일 없음" in n and "fetch_public_data.py" in n and "휴게음식점" in n for n in r.plan.notes)
    r2 = analyze("애월읍에 카페를 내면 어때?", s, make_provider("offline"))
    notes = " ".join(r2.plan.notes)
    assert "도민 시간대별 유동인구" in notes and "BC카드" in notes
    assert all(x.error == "" for x in r2.steps)


def test_benchmark_uses_my_store_locally_only(store):
    r = analyze("노형동 한식집 매출이 동네 평균보다 어때?", store, make_provider("offline"))
    assert r.bench is not None and 0.5 < r.bench.ratio_vs_emd < 1.5
    local = [s for s in r.steps if s.step.source == "local"]
    assert local and local[0].df_export is None and local[0].gate is None
    assert "내 매출/동네 평균" in r.bench.llm_safe and not any(isinstance(v, (int, float)) and v > 100000 for v in r.bench.llm_safe.values())


def test_same_question_same_sql_and_same_result(store):
    a = analyze("애월읍에 카페를 내면 어때?", store, make_provider("offline"))
    b = analyze("애월읍에 카페를 내면 어때?", store, make_provider("offline"))
    assert [s.sql_run for s in a.steps] == [s.sql_run for s in b.steps]
    assert a.explanation.headline == b.explanation.headline
    for x, y in zip(a.steps, b.steps):
        assert x.df.astype(str).equals(y.df.astype(str))


def test_missing_industry_asks_for_confirmation(store):
    r = None
    from jangsa.agents.planner import build_plan
    from jangsa.runlog import RunLog
    plan = build_plan("애월읍 치킨집 어때?", store, make_provider("offline"), RunLog())
    assert plan.needs_confirm == ["industry"] and any("치킨" in n for n in plan.notes)
    plan2 = build_plan("애월읍 치킨집 어때?", store, make_provider("offline"), RunLog(), {"industry": "간이음식"})
    assert not plan2.needs_confirm and plan2.steps


def test_checklist_lists_center_tables_with_numbers(store):
    from jangsa.agents.planner import build_plan
    from jangsa.runlog import RunLog
    plan = build_plan("애월읍에 카페를 내면 어때?", store, make_provider("offline"), RunLog())
    nos = {c.center_no for c in plan.checklist}
    assert {"1-08", "1-14"} <= nos
    c = next(c for c in plan.checklist if c.center_no == "1-08")
    assert "I56220" in c.condition and c.kind == "center"


def test_report_has_attribution_synthetic_notice_png_and_no_raw_sums(store):
    r = analyze("애월읍에 카페를 내면 어때?", store, make_provider("offline"))
    html = build_report_html(r)
    assert ATTRIBUTION in html and "가상 데이터" in html and "data:image/png;base64," in html
    assert "<svg" not in html and "plotly" not in html                      # 반출 정책: 그림(PNG)만
    assert "<th>이용 건수</th>" not in html and "<th>평균 매장 수(곳)</th>" not in html     # 엄격 모드: 원본 불가 표의 근거표에 단순집계 열 없음
    assert "<th>매출 규모 지수" in html or "<th>sales_idx" in html or "지수" in html
