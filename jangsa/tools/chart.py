"""도구 4. chart: 그래프 그리기. 같은 ChartSpec 으로 화면(plotly)과 그림 파일(PNG, matplotlib)을 만든다.

반출 정책상 시각화는 그림 파일(png, jpg)만 가능하므로 보고서에는 PNG 를 넣는다.
그래프에는 export_gate 를 통과한(소수 셀이 가려진) 표만 넣는다.
"""
from __future__ import annotations

import base64
import io
import sys
from dataclasses import dataclass, field

import pandas as pd

BASE_COLOR = "#5B7FA6"
HILITE_COLOR = "#E4572E"
LINE_COLORS = ["#2F6DB5", "#E4572E", "#3C9D6B", "#8A6FBF"]


@dataclass
class ChartSpec:
    kind: str                       # bar | hbar | line
    x: str
    y: list[str]
    title: str = ""
    x_title: str = ""
    y_title: str = ""
    highlight: str | None = None    # x 값 하나를 강조(내가 고른 읍면동 등)
    scale: float = 1.0              # y 를 이 값으로 나눠 그린다(예: 1e6 → 백만 원)
    top_n: int | None = None        # hbar/bar 에서 위에서 n 개만(강조 항목은 항상 포함)
    sort_desc: bool = False         # 값 기준으로 정렬할지
    ascending: bool = False         # sort_desc 가 True 일 때 작은 값부터
    labels: list[str] = field(default_factory=list)   # y 열의 범례 이름
    note: str = ""                  # 그래프 아래 작은 글씨


def prepare(spec: ChartSpec, df: pd.DataFrame) -> pd.DataFrame:
    d = df[[spec.x] + spec.y].copy()
    for c in spec.y:
        d[c] = pd.to_numeric(d[c], errors="coerce") / spec.scale
    if spec.kind in ("bar", "hbar") and spec.sort_desc:
        d = d.sort_values(spec.y[0], ascending=spec.ascending, na_position="last")
    if spec.top_n and len(d) > spec.top_n:
        top = d.head(spec.top_n)
        if spec.highlight is not None and spec.highlight not in set(top[spec.x]):
            hi = d[d[spec.x] == spec.highlight]
            top = pd.concat([top, hi])
        d = top
    return d.reset_index(drop=True)


def to_plotly(spec: ChartSpec, df: pd.DataFrame):
    import plotly.graph_objects as go
    d = prepare(spec, df)
    fig = go.Figure()
    names = spec.labels or spec.y
    if spec.kind in ("bar", "hbar"):
        colors = [HILITE_COLOR if (spec.highlight is not None and v == spec.highlight) else BASE_COLOR for v in d[spec.x]]
        horizontal = spec.kind == "hbar"
        dd = d.iloc[::-1] if horizontal else d
        cc = colors[::-1] if horizontal else colors
        vals = dd[spec.y[0]]
        text = [("비공개" if pd.isna(v) else f"{v:,.1f}".rstrip("0").rstrip(".")) for v in vals]
        fig.add_trace(go.Bar(x=vals if horizontal else dd[spec.x], y=dd[spec.x] if horizontal else vals,
                             orientation="h" if horizontal else "v", marker_color=cc, text=text, textposition="outside",
                             cliponaxis=False, name=names[0]))
    else:
        for i, c in enumerate(spec.y):
            fig.add_trace(go.Scatter(x=d[spec.x], y=d[c], mode="lines+markers", name=names[i],
                                     line=dict(color=LINE_COLORS[i % 4], width=2.5)))
    fig.update_layout(title=dict(text=spec.title, x=0, font=dict(size=16)), margin=dict(l=10, r=30, t=50, b=10),
                      height=max(320, 26 * len(d) + 120) if spec.kind == "hbar" else 340,
                      xaxis_title=spec.y_title if spec.kind == "hbar" else spec.x_title,
                      yaxis_title=spec.x_title if spec.kind == "hbar" else spec.y_title,
                      showlegend=len(spec.y) > 1, template="plotly_white",
                      font=dict(family="Malgun Gothic, Apple SD Gothic Neo, Noto Sans KR, sans-serif"),
                      legend=dict(orientation="h", y=1.1))
    return fig


def _set_font() -> None:
    from matplotlib import font_manager
    names = {f.name for f in font_manager.fontManager.ttflist}
    for cand in ("Malgun Gothic", "AppleGothic", "Apple SD Gothic Neo", "NanumGothic", "Noto Sans CJK KR", "Noto Sans KR"):
        if cand in names:
            import matplotlib
            matplotlib.rcParams["font.family"] = cand
            break
    import matplotlib
    matplotlib.rcParams["axes.unicode_minus"] = False


def to_png(spec: ChartSpec, df: pd.DataFrame, dpi: int = 140) -> bytes:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _set_font()
    d = prepare(spec, df)
    horizontal = spec.kind == "hbar"
    h = max(3.0, 0.27 * len(d) + 1.2) if horizontal else 3.4
    fig, ax = plt.subplots(figsize=(7.4, h))
    names = spec.labels or spec.y
    if spec.kind in ("bar", "hbar"):
        colors = [HILITE_COLOR if (spec.highlight is not None and v == spec.highlight) else BASE_COLOR for v in d[spec.x]]
        vals = d[spec.y[0]].fillna(0)
        if horizontal:
            ypos = range(len(d))
            ax.barh(list(ypos), vals, color=colors)
            ax.set_yticks(list(ypos), d[spec.x])
            ax.invert_yaxis()
            for i, v in enumerate(d[spec.y[0]]):
                ax.text(max(vals) * 0.01 if pd.isna(v) else v, i, "비공개" if pd.isna(v) else f" {v:,.1f}".rstrip("0").rstrip("."),
                        va="center", fontsize=8)
            ax.set_xlabel(spec.y_title)
        else:
            ax.bar(d[spec.x].astype(str), vals, color=colors)
            for i, v in enumerate(d[spec.y[0]]):
                ax.text(i, 0 if pd.isna(v) else v, "비공개" if pd.isna(v) else f"{v:,.1f}".rstrip("0").rstrip("."),
                        ha="center", va="bottom", fontsize=8)
            ax.set_ylabel(spec.y_title)
            ax.set_xlabel(spec.x_title)
    else:
        for i, c in enumerate(spec.y):
            ax.plot(d[spec.x].astype(str), d[c], marker="o", color=LINE_COLORS[i % 4], label=names[i], linewidth=2)
        ax.set_ylabel(spec.y_title)
        ax.set_xlabel(spec.x_title)
        step = max(1, len(d) // 8)
        ax.set_xticks(range(0, len(d), step), [str(v) for v in d[spec.x].iloc[::step]], rotation=30, ha="right", fontsize=8)
        if len(spec.y) > 1:
            ax.legend(frameon=False, fontsize=8)
    ax.set_title(spec.title, loc="left", fontsize=11)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x" if horizontal else "y", alpha=0.25)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    return buf.getvalue()


def to_png_base64(spec: ChartSpec, df: pd.DataFrame) -> str:
    return base64.b64encode(to_png(spec, df)).decode("ascii")
