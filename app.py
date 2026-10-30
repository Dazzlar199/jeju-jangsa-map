"""우리 동네 장사 지도 — 화면(Streamlit). 실행: run.bat / run.sh  (streamlit run app.py)"""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import streamlit as st

from jangsa.agents.explainer import explainer_messages
from jangsa.agents.planner import PLANNER_SYSTEM, build_plan, planner_messages
from jangsa.agents.rules import INTENTS, parse_question
from jangsa.config import ATTRIBUTION, DEMO, REPORTS_DIR, SYNTHETIC_NOTICE, llm_config
from jangsa.fmt import INTENT_KO, ORIGIN_KO, pretty_df
from jangsa.llm import LLMError, estimate_tokens, load_local_env, make_provider
from jangsa.pipeline import AnalysisResult, display_numbers, display_path, run_plan
from jangsa.report import build_report_html, chart_spec_from
from jangsa.runlog import RunLog
from jangsa.schema import EMD_LIST, INDUSTRIES, TABLES
from jangsa.store import DataStore, ensure_synthetic, read_public, read_synthetic
from jangsa.tools.chart import to_plotly, to_png
from jangsa.tools.csv_profile import profile_csv

st.set_page_config(page_title="우리 동네 장사 지도", page_icon="🗺️", layout="wide")

EXAMPLES = [
    "애월읍에 카페를 내면 어때?",
    "노형동 한식집 매출이 동네 평균보다 어때?",
    "성산읍에 카페가 너무 많은가요?",
    "중문동 카페는 몇 시에 제일 잘 돼?",
    "구좌읍 일식집은 비 오는 달에 매출이 줄어?",
    "제주시에서 편의점 하기 좋은 동네는?",
]
NONE_EMD = "(정하지 않음)"
CFG = llm_config()

st.markdown("""
<style>
.block-container {padding-top: 1.6rem; max-width: 1200px;}
.headline {background:#eef4fb; border-left:6px solid #5B7FA6; padding:14px 16px; border-radius:6px; font-size:1.15rem; font-weight:600; margin:8px 0 14px;}
.badge {display:inline-block; padding:2px 9px; border-radius:10px; font-size:0.78rem; font-weight:600; margin-right:6px;}
.b-syn {background:#fff3cd; color:#7a5c00; border:1px solid #e0c36a;}
.b-pub {background:#dff3e4; color:#1d6b3a; border:1px solid #8fcaa1;}
.b-user {background:#e8e4f5; color:#4a3a8a; border:1px solid #b6aadf;}
.small {color:#5a6b7b; font-size:0.85rem;}
.foot {margin-top:36px; padding-top:10px; border-top:1px solid #d5dde6; color:#5a6b7b; font-size:0.82rem;}
</style>
""", unsafe_allow_html=True)


# ── 상태 ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _download_public():
    """Cloud 데모만: 저장소에 없는 공개 파일을 첫 실행 때 받는다(서버가 켜져 있는 동안 한 번). 실패하면 기존 '파일 없음' 안내."""
    from jangsa.fetch_public import fetch_missing
    return fetch_missing()


@st.cache_resource(show_spinner=False)
def _base_frames():
    if DEMO:
        _download_public()
    return read_synthetic(include_my_store=False), read_public()


def _new_store() -> DataStore:
    syn, pub = _base_frames()
    s = DataStore()
    for r in syn:
        s.register(r, origin="synthetic")
    for p in pub:
        s.register_public(p)
    return s


def S():
    ss = st.session_state
    if "store" not in ss:
        with st.spinner(("가상 데이터를 만들고 공개 실데이터(휴게음식점·BC카드)를 공개 주소에서 받는 중입니다(처음 한 번만, 받기는 요청마다 20초까지 기다립니다)…"
                      if DEMO else "가상 데이터와 공개 실데이터를 불러오는 중입니다(처음 한 번만 오래 걸립니다)…")):
            ss.store = _new_store()
        ss.update(question="", plan=None, result=None, log=None, uploaded={}, confirm={}, report_path=None, saved_reports=[], auto_run=False)
    return ss


ss = S()
store: DataStore = ss.store


def badge(origin: str) -> str:
    cls = {"synthetic": "b-syn", "public": "b-pub", "user": "b-user"}.get(origin, "b-pub")
    return f"<span class='badge {cls}'>{ORIGIN_KO.get(origin, origin)}</span>"


# ── 사이드바: 분석 방식 · 데이터 · 삭제 ──────────────────────────────────────
with st.sidebar:
    st.header("1. 분석 방식")
    labels = {k: v["label"] for k, v in CFG["providers"].items()}
    pname = st.radio("어떤 방식으로 질문을 풀까요?", list(labels), format_func=lambda k: labels[k], index=0,
                     help="오프라인 모드는 키·인터넷 없이 규칙과 템플릿으로 끝까지 돌아갑니다. LLM 을 고르면 계획·해설 문장을 더 유연하게 만듭니다.")
    api_key, model = "", CFG["providers"][pname]["model"]
    if pname == "offline":
        st.success("오프라인 모드: 외부로 보내는 내용이 전혀 없습니다. 비용 0원." + (" (공개 데모는 이 모드가 기본입니다.)" if DEMO else ""))
    else:
        env_has = bool(load_local_env().get(CFG["providers"][pname]["key_env"]))
        api_key = st.text_input("API 키", type="password", help="화면에만 쓰이고 저장되지 않습니다. 비워 두면 .env 파일의 키를 씁니다.")
        st.caption("키 상태: " + ("입력됨" if api_key else (".env 에서 찾음" if env_has else "없음 → 키를 넣어 주세요")))
        model = st.text_input("모델 이름", value=CFG["providers"][pname]["model"])
        st.caption("키는 이 브라우저 세션의 메모리에만 있고 서버에 저장하지 않습니다(탭을 닫으면 사라짐)."
                   + (" 공개 데모에서는 본인 키를 넣는 것을 권하지 않습니다. 오프라인 모드로 충분히 체험할 수 있습니다." if DEMO else ""))
        st.caption(f"질문 하나당 LLM 호출 최대 {CFG['call_limit_per_question']}회. LLM 에는 표 설명(열 이름)과 질문, 반출 검사를 통과한 집계표만 갑니다.")
    provider = make_provider(pname, model, api_key or None)

    st.header("2. 데이터")
    infos = [i for i in store.tables.values()]
    n_syn = sum(1 for i in infos if i.origin == "synthetic")
    if n_syn:
        st.warning(f"가상 데이터 {n_syn}종을 쓰는 중입니다(실제 통계 아님).")
    with st.expander(f"불러온 표 {len(infos)}종", expanded=False):
        for i in infos:
            per = f" · {i.period[0]}~{i.period[1]}" if i.period else ""
            st.markdown(f"{badge(i.origin)} **{i.table_id}** {i.rows:,}행{per}", unsafe_allow_html=True)
            st.caption(i.file_name)
            for n in i.notes[:2]:
                st.caption("· " + n)

    st.markdown("**센터에서 받은 CSV 올리기**")
    st.caption("센터 PC 에서 내려받은 CSV(CP949·UTF-8 모두 가능)를 올리면 열을 자동으로 맞춥니다. 올리는 즉시 가상 데이터 중 같은 표가 바뀝니다.")
    files = st.file_uploader("CSV 파일(여러 개 가능)", type=["csv"], accept_multiple_files=True, key="center_up")
    for f in files or []:
        key = f"{f.name}:{f.size}"
        raw = f.getvalue()
        res = profile_csv(raw, f.name)
        if res.ok and res.table_id not in ("my_store",) and key not in ss.uploaded:
            store.register(res, origin="center")
            ss.uploaded[key] = res.table_id
            ss.result = None
        icon = "✅" if res.ok else "⚠️"
        with st.expander(f"{icon} {f.name}", expanded=not res.ok):
            if res.table_id:
                spec = TABLES[res.table_id]
                st.write(f"표 종류: **{spec.center_no} {spec.center_name}**  · {res.encoding} · {res.rows:,}행")
                st.write("열 맞춤: " + ", ".join(f"{k} ← {v}" for k, v in res.mapping.items()))
            for e in res.errors:
                st.error(e)
            for w in res.warnings:
                st.warning(w)
            if not res.ok:
                cands = [t for t, sp in TABLES.items() if sp.synthetic_file and t != "my_store"]
                tid = st.selectbox("표 종류를 직접 고르기", cands, index=cands.index(res.table_id) if res.table_id in cands else 0,
                                   format_func=lambda t: f"{TABLES[t].center_no} {TABLES[t].center_name}", key=f"tid_{key}")
                cols = [""] + (res.unmapped_source_columns or list(profile_csv(raw, f.name, table_id=tid).unmapped_source_columns))
                manual = {}
                for c in TABLES[tid].read_columns:
                    if c.required and c.name not in res.mapping:
                        v = st.selectbox(f"'{c.name}'({c.desc}) 열", cols, key=f"m_{key}_{c.name}")
                        if v:
                            manual[c.name] = v
                if st.button("이 설정으로 다시 맞추기", key=f"retry_{key}"):
                    r2 = profile_csv(raw, f.name, table_id=tid, manual_mapping=manual)
                    if r2.ok:
                        store.register(r2, origin="center")
                        ss.uploaded[key] = tid
                        ss.result = None
                        st.rerun()
                    else:
                        st.error("; ".join(r2.errors))

    st.markdown("**내 가게 월 매출 CSV (선택)**")
    st.caption("두 열(연월, 매출)이면 됩니다. 이 자료는 이 PC 화면 안에서만 쓰고 LLM 으로 보내지 않습니다.")
    mine = st.file_uploader("내 가게 CSV", type=["csv"], key="my_up")
    if mine is not None:
        r = profile_csv(mine.getvalue(), mine.name, table_id="my_store")
        if r.ok:
            store.register(r, origin="user")
            st.success(f"내 가게 {r.rows}개월 매출을 불러왔습니다.")
        else:
            st.error("; ".join(r.errors))
    elif st.checkbox("가상 내 가게 예시 쓰기(노형동 한식집, 가상)", value=False, key="demo_mine"):
        r = profile_csv(ensure_synthetic() / TABLES["my_store"].synthetic_file, "가상_내가게_월매출.csv", table_id="my_store")
        if r.ok:
            store.register(r, origin="synthetic")

    st.header("3. 개인 자료 삭제")
    st.caption("내 가게 CSV·센터에서 올린 CSV 는 이 화면의 메모리에만 있고 디스크에 쓰지 않습니다. 분석이 끝나면 지우세요.")
    if st.button("🗑️ 올린 자료 모두 삭제", type="secondary"):
        store.clear("user")
        store.clear("center")
        for p in ss.saved_reports:
            Path(p).unlink(missing_ok=True)
        ss.update(result=None, plan=None, uploaded={}, saved_reports=[], report_path=None)
        st.success("올린 자료와 이번에 저장한 보고서를 지웠습니다.")
        st.rerun()


# ── 본문 ────────────────────────────────────────────────────────────────
st.title("🗺️ 우리 동네 장사 지도")
st.markdown("**제주에서 가게를 열려는 분, 매출이 고민인 분**을 위한 동네 장사 진단입니다. 궁금한 것을 한 문장으로 물어보세요.")
if any(i.origin == "synthetic" for i in store.tables.values()):
    st.info(SYNTHETIC_NOTICE + " 센터에서 받은 CSV 를 올리면 같은 화면으로 실제 분석이 됩니다. "
            "**실데이터** 표시가 붙은 단계(관광객 시간대, 도민 유동인구, 서귀포 휴게음식점, 버스 정류소)만 공개 실데이터이며, 해당 파일이 `data/public/` 에 있을 때만 나옵니다.")

st.subheader("① 무엇이 궁금하세요?")
cols = st.columns(3)
for i, ex in enumerate(EXAMPLES):
    if cols[i % 3].button(ex, key=f"ex{i}", width="stretch"):
        ss.question = ex
        ss.auto_run = True
        ss.confirm = {}
q = st.text_input("질문", value=ss.question, placeholder="예) 애월읍에 카페를 내면 어때?", label_visibility="collapsed")
if q != ss.question:
    ss.question = q
    ss.confirm = {}
st.caption("개인정보(이름·전화번호·주소)는 질문에 쓰지 마세요. 질문 글은 LLM 모드에서 그대로 전송됩니다.")
go = st.button("질문하기", type="primary") or (ss.auto_run and ss.question)

if go and ss.question.strip():
    ss.auto_run = False
    ss.log = RunLog()
    with st.spinner("질문을 읽고 계획을 세우는 중…"):
        ss.plan = build_plan(ss.question, store, provider, ss.log, ss.confirm or None)
    ss.result = None
    if not ss.plan.needs_confirm and ss.plan.steps:
        with st.spinner("분석하는 중…"):
            try:
                ss.result = run_plan(ss.plan, store, provider, ss.log)
            except (LLMError, ValueError) as e:
                st.error(f"분석 중 문제가 생겼습니다: {e}")

plan = ss.plan
if plan is not None:
    st.subheader("② 이렇게 이해했어요")
    c1, c2, c3 = st.columns([2, 2, 2])
    c1.markdown(f"**무엇을 볼까요**: {INTENT_KO.get(plan.intent, plan.intent)}  \n<span class='small'>{plan.intent_label}</span>", unsafe_allow_html=True)
    emd_opts = [NONE_EMD] + EMD_LIST
    emd_sel = c2.selectbox("동네(읍면동)", emd_opts, index=emd_opts.index(plan.emd) if plan.emd in emd_opts else 0)
    ind_opts = ["(골라 주세요)"] + INDUSTRIES
    ind_sel = c3.selectbox("업종", ind_opts, index=ind_opts.index(plan.industry) if plan.industry in ind_opts else 0)
    for n in plan.notes:
        st.caption("· " + n)
    new_conf = {"emd": None if emd_sel == NONE_EMD else emd_sel, "industry": None if ind_sel == ind_opts[0] else ind_sel}
    if (new_conf["emd"] != plan.emd or new_conf["industry"] != plan.industry):
        if st.button("이 조건으로 다시 분석", type="primary"):
            ss.confirm = {**new_conf, "intent": plan.intent}
            ss.auto_run = True
            st.rerun()
    if plan.needs_confirm:
        need = {"industry": "업종", "emd": "동네"}
        st.warning("먼저 " + "·".join(need[n] for n in plan.needs_confirm) + "을(를) 골라 주세요. 고른 뒤 '이 조건으로 다시 분석'을 누르면 시작합니다.")

    with st.expander("📋 센터에서 받아 올 CSV 체크리스트", expanded=bool(plan.needs_confirm or any(c.status == "없음" and c.kind == "center" for c in plan.checklist))):
        st.caption("센터 PC 에서 아래 표를 CSV 로 내려받아 왼쪽 '센터에서 받은 CSV 올리기'에 올리세요. (센터 데이터 시스템과 직접 연결되지 않습니다)")
        rows = []
        kind_ko = {"center": "센터 CSV", "optional": "직접 만든 집계(없어도 됨)", "public": "공개 실데이터(data/public/)", "user": "내 자료(선택)"}
        for c in plan.checklist:
            rows.append({"표 번호": c.center_no, "표 이름": c.center_name, "받을 조건": c.condition, "구분": kind_ko.get(c.kind, c.kind),
                         "상태": ("✅ " + ORIGIN_KO.get(c.origin, c.origin)) if c.status == "불러옴"
                         else ("❌ 파일 없음 — 받는 법: scripts/fetch_public_data.py" if c.kind == "public" else "❌ 없음")})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    with st.expander("🔍 LLM 에 보내는 내용 미리보기"):
        parsed = parse_question(plan.question)
        sys_, usr = planner_messages(plan.question, parsed, plan.period, plan.period_all, [t for t in store.available() if t != "my_store"])
        if provider.is_offline:
            st.success("지금은 오프라인 모드라 **아무것도 전송되지 않습니다**. LLM 모드를 고르면 아래 두 덩어리가 계획용으로 갑니다.")
        else:
            st.warning(f"{CFG['providers'][pname]['label']} 로 아래 내용이 갑니다. 실제 데이터 값은 들어 있지 않습니다(표 설명과 질문만).")
        st.caption(f"[계획 에이전트] 약 {estimate_tokens(sys_ + usr):,} 토큰")
        st.text_area("시스템 지침", sys_, height=140, disabled=True)
        st.text_area("사용자 메시지(질문 + 표 명세)", usr, height=220, disabled=True)
        if ss.result:
            res0 = ss.result
            es, eu = explainer_messages(plan.question, res0.plan, [r for r in res0.steps if r.step.source != "local"], res0.bench,
                                        "가상(시연용) 데이터" if res0.synthetic else "센터·공개 데이터", res0.evidence_no)
            st.caption(f"[해설 에이전트] 반출 검사를 통과한 집계표만 · 약 {estimate_tokens(es + eu):,} 토큰 (개인 자료의 금액은 제외, 엄격 모드면 단순집계 제외)")
            st.text_area("해설 에이전트에 보낼 내용", eu, height=260, disabled=True)

res: AnalysisResult | None = ss.result
if res is not None:
    plan, ex, nums = res.plan, res.explanation, display_numbers(res)
    st.subheader("③ 결과")
    st.markdown(f"<div class='headline'>{ex.headline}</div>", unsafe_allow_html=True)
    for n in ex.notes + plan.notes:
        st.caption("· " + n)
    st.markdown("**쉬운 말 요약**")
    for p in ex.points:
        ref = f" <span class='small'>[근거표 {', '.join(map(str, p['tables']))}]</span>" if p.get("tables") else ""
        st.markdown(f"- {p['text']}{ref}", unsafe_allow_html=True)
    if ex.checkpoints:
        st.markdown("**지금 확인할 것**")
        for c in ex.checkpoints:
            st.markdown(f"- {c}")
    st.markdown("---")
    shown = [r for r in res.steps if not r.skipped]
    for sr in shown:
        n = nums.get(sr.step.id)
        st.markdown(f"#### 근거표 {n or '-'}. {sr.step.title}", unsafe_allow_html=True)
        st.markdown(badge(sr.origin) + f"<span class='small'>{sr.step.purpose}</span>", unsafe_allow_html=True)
        if sr.error:
            st.error(f"실행하지 못했습니다: {sr.error}")
            continue
        if sr.gate and not sr.gate.passed:
            st.error("🔒 반출 기준 검사에서 막혀 수치를 보여 드리지 않습니다: " + "; ".join(sr.gate.blocked_reasons))
            continue
        d = sr.df
        left, right = st.columns([3, 2])
        spec = chart_spec_from(sr.step.chart, d) if sr.step.chart else None
        if spec:
            left.plotly_chart(to_plotly(spec, d), width="stretch")
        elif d is not None:
            left.write("")
        with right:
            with st.expander("근거표 보기", expanded=spec is None):
                st.dataframe(pretty_df(d), hide_index=True, width="stretch")
            if sr.gate:
                tag = "통과" if not sr.gate.suppressed_rows else f"통과(소수 셀 {sr.gate.suppressed_rows}행 '비공개' 처리)"
                st.caption(f"반출 검사: {tag}")
                if sr.gate.dropped_for_export:
                    st.caption("엄격 모드: 합계·건수 열은 이 화면에서만 보이고, LLM·보고서에는 지수·비율·순위만 갑니다.")
                if "csv" in sr.gate.allowed_formats and d is not None and sr.step.source != "local":
                    st.download_button("표를 CSV 로 저장(반출 가능한 표)", d.to_csv(index=False).encode("utf-8-sig"),
                                       file_name=f"{sr.step.id}.csv", mime="text/csv", key=f"csv_{sr.step.id}")
            if spec:
                dd = sr.df_export if (sr.df_export is not None and sr.step.source != "local") else d
                if dd is not None and all(c in dd.columns for c in [spec.x] + spec.y):
                    st.download_button("그래프를 PNG 로 저장", to_png(spec, dd), file_name=f"{sr.step.id}.png", mime="image/png", key=f"png_{sr.step.id}")
    for sr in res.steps:
        if sr.skipped.startswith("[실데이터]"):
            st.info(f"{sr.step.title}\n\n{sr.skipped}")
    st.markdown("**한계**")
    for l in ex.limits:
        st.markdown(f"- {l}")

    st.markdown("---")
    st.subheader("④ 보고서")
    html = build_report_html(res)
    c1, c2 = st.columns(2)
    c1.download_button("📄 보고서 HTML 내려받기(1~2쪽, 그래프는 PNG)", html.encode("utf-8"), file_name="장사지도_보고서.html", mime="text/html")
    if c2.button("보고서를 reports/ 폴더에 저장"):
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        from datetime import datetime
        p = REPORTS_DIR / f"report_{datetime.now():%Y%m%d-%H%M%S}.html"
        p.write_text(html, encoding="utf-8")
        ss.saved_reports.append(str(p))
        st.success(f"저장했습니다: {display_path(p)}")
    st.caption("센터 밖으로 가져가려면 센터의 반출 신청·심사가 필요합니다. 그림(PNG)·분석결과보고서 형태를 권장합니다.")

    st.subheader("⑤ 분석 기록")
    log = res.log
    with st.expander("실행한 SQL · 사용 파일 · 반출 검사 · 호출 수와 비용", expanded=False):
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("LLM 호출", f"{log.n_calls}회")
        m2.metric("토큰(입력/출력)", f"{log.total_in:,} / {log.total_out:,}")
        m3.metric("예상 비용", f"${log.total_cost_usd:.4f}" + (f" (약 {log.total_cost_usd * 1400:,.0f}원)" if log.total_cost_usd else ""))
        m4.metric("모드", f"{log.mode}")
        st.markdown("**사용한 파일**")
        st.dataframe(pd.DataFrame([{"표": f["table"], "센터 표 번호": f["center_no"], "파일": f["file"], "행 수": f["rows"],
                                    "구분": ORIGIN_KO.get(f["origin"], f["origin"]),
                                    "기간": f"{f['period'][0]}~{f['period'][1]}" if f.get("period") else ""} for f in res.files]),
                     hide_index=True, width="stretch")
        st.markdown("**단계별 SQL**")
        for s in log.steps:
            st.markdown(f"**{s['title']}**  \n<span class='small'>{s['source']} · {s['rows']}행(원천 {s['source_rows']:,}행) · 시도 {s['attempts']}회 · "
                        f"반출 검사: {s['gate'] or s['skipped'] or s['error']} · {s['ms']}ms</span>", unsafe_allow_html=True)
            st.code(s["sql"], language="sql")
        if log.llm_calls:
            st.markdown("**LLM 호출 내역**")
            st.dataframe(pd.DataFrame([c.__dict__ for c in log.llm_calls]), hide_index=True, width="stretch")
        if res.log_path:
            st.caption(f"기록 파일(데이터 값 없음): {res.log_path}")

st.markdown(f"<div class='foot'>{ATTRIBUTION}.<br>가상 데이터는 시연용이며 실제 제주의 통계가 아닙니다. "
            f"[실데이터] 표시는 제주데이터허브·공공데이터포털의 공개 파일입니다.</div>", unsafe_allow_html=True)
