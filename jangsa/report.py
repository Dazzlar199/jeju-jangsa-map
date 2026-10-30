"""보고서 HTML(1~2쪽, A4 인쇄용). 그래프는 PNG 그림으로 넣는다(반출 정책: 시각화는 그림 파일).

들어가는 것: 질문·결론, 쉬운 말 요약(근거표 번호), 그래프, 근거표, 한계, 분석 기록 요약(SQL·사용 파일·반출 검사),
출처 문구. 가상 데이터를 쓴 경우 맨 위에 표시한다.
"""
from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path

import pandas as pd

from .config import ATTRIBUTION, REPORTS_DIR, SYNTHETIC_NOTICE
from .fmt import INTENT_KO, ORIGIN_KO, pretty_df
from .pipeline import AnalysisResult, display_numbers
from .tools.chart import ChartSpec, to_png_base64

CSS = """
@page { size: A4; margin: 14mm; }
body { font-family: 'Malgun Gothic','Apple SD Gothic Neo','Noto Sans KR',sans-serif; color:#1b2733; font-size:10.5pt; line-height:1.5; margin:0 auto; max-width:820px; padding:12px; }
h1 { font-size:18pt; margin:0 0 4px; } h2 { font-size:12.5pt; margin:16px 0 6px; border-bottom:2px solid #5B7FA6; padding-bottom:2px; }
.meta { color:#5a6b7b; font-size:9.5pt; } .badge { display:inline-block; background:#fff3cd; border:1px solid #e0c36a; padding:3px 8px; border-radius:6px; font-size:9.5pt; margin:6px 0; }
.headline { background:#eef4fb; border-left:5px solid #5B7FA6; padding:10px 12px; font-size:12pt; font-weight:600; margin:8px 0; }
ul { margin:4px 0 4px 18px; padding:0; } li { margin:2px 0; } .ref { color:#5B7FA6; font-size:9pt; }
.charts { display:grid; grid-template-columns:1fr 1fr; gap:8px; } .charts figure { margin:0; break-inside:avoid; } .charts img { width:100%; }
table { border-collapse:collapse; width:100%; font-size:8.5pt; margin:4px 0 8px; } th,td { border:1px solid #cfd8e3; padding:2px 5px; text-align:right; } th { background:#f1f5f9; } td:first-child, th:first-child { text-align:left; }
.cap { font-size:9pt; font-weight:600; margin-top:8px; } pre { background:#f6f8fa; padding:6px; font-size:7.5pt; white-space:pre-wrap; word-break:break-all; margin:2px 0 6px; }
footer { margin-top:16px; border-top:1px solid #cfd8e3; padding-top:6px; font-size:8.5pt; color:#5a6b7b; }
.note { font-size:8.5pt; color:#7a5c00; }
"""


def chart_spec_from(step_chart: dict, df: pd.DataFrame) -> ChartSpec | None:
    if not step_chart or df is None or df.empty:
        return None
    try:
        keys = {k: v for k, v in step_chart.items() if k in ChartSpec.__dataclass_fields__}
        spec = ChartSpec(**keys)
        if spec.x not in df.columns or any(y not in df.columns for y in spec.y):
            return None
        return spec
    except TypeError:
        return None


def _esc(s) -> str:
    return html.escape(str(s))


def _table_html(df: pd.DataFrame, n: int) -> str:
    p = pretty_df(df, 14)
    head = "".join(f"<th>{_esc(c)}</th>" for c in p.columns)
    body = "".join("<tr>" + "".join(f"<td>{_esc(v)}</td>" for v in row) + "</tr>" for row in p.itertuples(index=False))
    more = f"<div class='note'>상위 {len(p)}행만 표시(전체 {len(df)}행)</div>" if len(df) > len(p) else ""
    return f"<table><tr>{head}</tr>{body}</table>{more}"


def build_report_html(res: AnalysisResult, with_sql: bool = True) -> str:
    plan, ex = res.plan, res.explanation
    nums = display_numbers(res)
    parts = [f"<h1>우리 동네 장사 지도 — 분석 보고서</h1>",
             f"<div class='meta'>{datetime.now():%Y-%m-%d %H:%M} 생성 · 질문: “{_esc(plan.question)}” · "
             f"{_esc(INTENT_KO.get(plan.intent, plan.intent))} · {_esc(plan.emd or plan.city or '제주 전체')} · {_esc(plan.industry or '')}</div>"]
    if res.synthetic:
        parts.append(f"<div class='badge'>⚠ {_esc(SYNTHETIC_NOTICE)}</div>")
    parts.append(f"<div class='headline'>{_esc(ex.headline)}</div>")
    parts.append("<h2>쉬운 말 요약</h2><ul>" + "".join(
        f"<li>{_esc(p['text'])} " + (f"<span class='ref'>[근거표 {', '.join(map(str, p['tables']))}]</span>" if p.get("tables") else "") + "</li>"
        for p in ex.points) + "</ul>")
    if ex.checkpoints:
        parts.append("<h2>지금 확인할 것</h2><ul>" + "".join(f"<li>{_esc(c)}</li>" for c in ex.checkpoints) + "</ul>")
    figs, tabs = [], []
    for sr in res.steps:
        d = sr.df_export if sr.step.source != "local" else sr.df     # 보고서는 반출용 표(엄격 모드면 응용집계만)를 쓴다
        if d is None:
            continue
        n = nums.get(sr.step.id)
        spec = chart_spec_from(sr.step.chart, d)
        if spec:
            figs.append(f"<figure><img alt='{_esc(sr.step.title)}' src='data:image/png;base64,{to_png_base64(spec, d)}'>"
                        f"<figcaption class='ref'>근거표 {n} · {_esc(ORIGIN_KO.get(sr.origin, sr.origin))}</figcaption></figure>")
        tabs.append(f"<div class='cap'>근거표 {n}. {_esc(sr.step.title)} <span class='ref'>({_esc(ORIGIN_KO.get(sr.origin, sr.origin))})</span></div>"
                    + _table_html(d, n or 0))
    if figs:
        parts.append("<h2>그래프</h2><div class='charts'>" + "".join(figs) + "</div>")
    if tabs:
        parts.append("<h2>근거표</h2>" + "".join(tabs))
    parts.append("<h2>한계</h2><ul>" + "".join(f"<li>{_esc(l)}</li>" for l in ex.limits) + "</ul>")
    # 분석 기록
    log = res.log
    rows = []
    for s in log.steps:
        sql = f"<pre>{_esc(s['sql'])}</pre>" if with_sql else ""
        rows.append(f"<div class='cap'>{_esc(s['title'])} <span class='ref'>({_esc(s['source'])}, {s['rows']}행, 반출 검사: {_esc(s['gate'] or s['skipped'] or s['error'])})</span></div>{sql}")
    files = "".join(f"<li>{_esc(f['table'])} · {_esc(f['file'])} · {f['rows']:,}행 · {_esc(ORIGIN_KO.get(f['origin'], f['origin']))}"
                    + (f" · {f['period'][0]}~{f['period'][1]}" if f.get("period") else "") + "</li>" for f in res.files)
    llm = (f"LLM 호출 {log.n_calls}회, 토큰 {log.total_in:,}/{log.total_out:,}, 예상 비용 약 ${log.total_cost_usd:.4f}"
           if log.n_calls else "LLM 호출 0회(오프라인 모드: 외부로 전송된 내용 없음)")
    parts.append(f"<h2>분석 기록</h2><div class='meta'>모드: {_esc(log.mode)} {_esc(log.model)} · {llm}</div>"
                 f"<ul>{files}</ul>{''.join(rows)}")
    parts.append("<div class='note'>이 보고서를 센터 밖으로 가져가려면 센터의 반출 신청·심사가 필요합니다. 그래프는 그림(PNG)으로, "
                 "표의 수치는 반출 기준(10 미만 가림)을 적용한 값입니다. 표 형식(csv·엑셀)으로의 반출은 제공처 기준에 따라 불가할 수 있습니다.</div>")
    parts.append(f"<footer>{_esc(ATTRIBUTION)}</footer>")
    return f"<!doctype html><html lang='ko'><head><meta charset='utf-8'><title>우리 동네 장사 지도 보고서</title><style>{CSS}</style></head><body>{''.join(parts)}</body></html>"


def save_report(res: AnalysisResult, folder: Path | str = REPORTS_DIR) -> Path:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    p = folder / f"report_{datetime.now():%Y%m%d-%H%M%S}.html"
    p.write_text(build_report_html(res), encoding="utf-8")
    return p
