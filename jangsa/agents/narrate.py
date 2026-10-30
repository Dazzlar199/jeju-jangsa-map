"""템플릿 해설: export_gate 를 통과한 표에서 숫자를 뽑아 쉬운 문장으로 쓴다(LLM 없이).

오프라인 모드의 해설이고, LLM 해설이 막히거나 검증에 실패했을 때의 대체 해설이기도 하다.
표에 없는 수치는 문장에 쓰지 않는다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ..schema import CODES


def josa(word: str, pair: str) -> str:
    """받침에 따라 조사를 붙인다. pair: '은/는' '을/를' '이/가' '으로/로'."""
    a, b = pair.split("/")
    ch = str(word).strip()[-1:]
    if ch and "가" <= ch <= "힣":
        code = (ord(ch) - 0xAC00) % 28
        has = code != 0
        if pair == "으로/로":
            return a if (has and code != 8) else b
        return a if has else b
    return b if pair != "으로/로" else b


def fmt_won(x: float) -> str:
    if x is None or pd.isna(x):
        return "비공개"
    x = float(x)
    if abs(x) >= 1e8:
        return f"{x / 1e8:,.1f}억 원"
    if abs(x) >= 1e4:
        return f"약 {round(x / 1e4):,}만 원"
    return f"{x:,.0f}원"


def fmt_num(x: float, nd: int = 1) -> str:
    if x is None or pd.isna(x):
        return "비공개"
    return f"{float(x):,.{nd}f}".rstrip("0").rstrip(".") if nd else f"{float(x):,.0f}"


@dataclass
class Finding:
    text: str
    signal: int | None = None       # +1 유리, 0 보통, -1 불리 (입지 점수판용)
    label: str = ""                 # 점수판 이름


def rank_of(df: pd.DataFrame, col: str, key: str, value, ascending: bool = False) -> tuple[int | None, int, float | None]:
    d = df.dropna(subset=[col])
    n = len(d)
    row = d[d[key] == value]
    if row.empty:
        return None, n, None
    v = float(row[col].iloc[0])
    r = int((d[col] < v).sum() if ascending else (d[col] > v).sum()) + 1
    return r, n, v


def _top(df: pd.DataFrame, col: str, key: str, k: int = 3, ascending: bool = False) -> list[str]:
    d = df.dropna(subset=[col]).sort_values(col, ascending=ascending).head(k)
    return list(d[key])


TB_HOURS = CODES["time_band_hours"]


def _pct(v) -> str:
    return fmt_num(v, 1)


def narrate_step(step_id: str, df: pd.DataFrame, emd: str | None, industry: str, ctx: dict | None = None) -> list[Finding]:
    """df 는 반출 가능한 표(df_export)다. 엄격 모드에서는 단순집계 열이 없으므로 응용집계 열(지수·순위·비율)만 쓴다."""
    ctx = ctx or {}
    if df is None or df.empty:
        return []
    who = emd or ctx.get("scope", "제주 전체")
    out: list[Finding] = []
    if step_id == "sales_by_emd":
        if emd:
            r, n, v = rank_of(df, "sales_idx", "emd", emd)
            if r is None:
                out.append(Finding(f"{emd}의 {industry} 카드매출은 소수 셀 기준에 걸려 공개할 수 없습니다(다른 동네와의 비교만 가능)."))
            else:
                row = df[df["emd"] == emd].iloc[0]
                tk = f" 결제 한 건당 평균은 약 {fmt_num(row['ticket_per_use'], 0)}원입니다." if "ticket_per_use" in df and pd.notna(row.get("ticket_per_use")) else ""
                sig = 1 if r <= n / 3 else (-1 if r > 2 * n / 3 else 0)
                out.append(Finding(f"{emd}의 {industry} 카드매출 규모는 읍면동 평균의 {_pct(v)}%로, 공개 가능한 {n}곳 중 {r}위입니다.{tk}", sig, "시장 규모"))
        else:
            top = _top(df, "sales_idx", "emd")
            out.append(Finding(f"{industry} 카드매출 규모가 큰 동네는 {', '.join(top)} 순입니다."))
    elif step_id == "sales_per_store_by_emd":
        if emd:
            r, n, v = rank_of(df, "sales_per_store", "emd", emd)
            if r is None:
                out.append(Finding(f"{emd}의 {industry} 매장당 매출은 소수 셀 기준으로 가려졌습니다."))
            else:
                mean = df["sales_per_store"].dropna().mean()
                sig = 1 if r <= n / 3 else (-1 if r > 2 * n / 3 else 0)
                out.append(Finding(f"{emd}의 {industry} 매장 한 곳당 월 카드매출은 {fmt_won(v)}{josa(fmt_won(v), '으로/로')}, 공개 가능한 {n}곳 중 {r}위입니다"
                                   f"(읍면동 평균 {fmt_won(mean)}).", sig, "매장당 매출"))
        else:
            top = _top(df, "sales_per_store", "emd")
            out.append(Finding(f"{industry} 매장 한 곳당 월 카드매출이 높은 동네는 {', '.join(top)} 순입니다."))
    elif step_id == "density_by_emd":
        if emd:
            r, n, v = rank_of(df, "stores_per_10k_pop", "emd", emd, ascending=True)
            if r is None:
                out.append(Finding(f"{emd}의 {industry} 경쟁 밀도는 소수 셀 기준으로 가려졌습니다."))
            else:
                sig = 1 if r <= n / 2 else (-1 if r > 0.75 * n else 0)
                tone = "경쟁이 덜한 편" if sig > 0 else ("경쟁이 치열한 편" if sig < 0 else "중간 수준")
                out.append(Finding(f"{emd}{josa(emd, '은/는')} 유동인구 1만 명당 {industry} 매장이 {fmt_num(v, 2)}곳으로, 적은 순 {r}/{n}번째라 {tone}입니다.", sig, "경쟁 밀도"))
        else:
            top = _top(df, "stores_per_10k_pop", "emd", ascending=True)
            out.append(Finding(f"유동인구 대비 {industry} 매장이 적어 여유가 있어 보이는 동네는 {', '.join(top)} 순입니다."))
    elif step_id == "closure_by_emd":
        if emd:
            r, n, v = rank_of(df, "close_rate_pct", "emd", emd, ascending=True)
            if r is None:
                out.append(Finding(f"{emd}의 {industry} 폐업 비율은 소수 셀 기준으로 가려졌습니다."))
            else:
                sig = 1 if r <= 0.4 * n else (-1 if r > 0.7 * n else 0)
                out.append(Finding(f"{emd}의 최근 12개월 {industry} 폐업 비율은 {fmt_num(v)}%로, 낮은 순 {r}/{n}번째입니다.", sig, "폐업 비율"))
        else:
            top = _top(df, "close_rate_pct", "emd", ascending=True)
            out.append(Finding(f"최근 12개월 {industry} 폐업 비율이 낮은 동네는 {', '.join(top)} 순입니다."))
    elif step_id == "time_band":
        d = df.dropna(subset=["sales_share_pct"]).sort_values("sales_share_pct", ascending=False)
        if len(d) >= 2:
            a, b = d.iloc[0], d.iloc[1]
            out.append(Finding(f"{who}의 {industry}{josa(industry, '은/는')} {a['time_band']}({TB_HOURS.get(a['time_band'], '')})에 매출의 {_pct(a['sales_share_pct'])}%, "
                               f"{b['time_band']}({TB_HOURS.get(b['time_band'], '')})에 {_pct(b['sales_share_pct'])}%가 몰립니다. 이 시간대에 인력과 재고를 맞추세요."))
    elif step_id == "dow":
        d = df.dropna(subset=["sales_share_pct"]).set_index("dow")["sales_share_pct"]
        wk = d.drop(index="공휴일", errors="ignore")
        if len(wk) >= 5:
            top = wk.sort_values(ascending=False)
            we = sum(d.get(x, 0) for x in ("토", "일"))
            out.append(Finding(f"{who}의 {industry}{josa(industry, '은/는')} {top.index[0]}요일 비중이 {_pct(top.iloc[0])}%로 가장 높고, 토·일을 합치면 {_pct(we)}%입니다."
                               + (f" 공휴일은 {_pct(d['공휴일'])}%입니다." if "공휴일" in d.index else "")))
    elif step_id == "monthly_sales":
        d = df.dropna(subset=["sales_idx"])
        if len(d) >= 6:
            hi, lo = d.loc[d["sales_idx"].idxmax()], d.loc[d["sales_idx"].idxmin()]
            out.append(Finding(f"{who}의 {industry} 월 카드매출은 {hi['ym']}에 가장 높고(월평균의 {_pct(hi['sales_idx'])}%) {lo['ym']}에 가장 낮아(월평균의 {_pct(lo['sales_idx'])}%) "
                               f"차이가 {hi['sales_idx'] / lo['sales_idx']:.1f}배입니다. 비수기를 버틸 운영자금을 따로 잡아 두세요."))
    elif step_id == "rain_effect":
        d = df.dropna(subset=["avg_sales_idx"]).set_index("rain_group")["avg_sales_idx"]
        if "비가 평소보다 많은 달" in d.index and "비가 평소보다 적은 달" in d.index:
            more, less = d["비가 평소보다 많은 달"], d["비가 평소보다 적은 달"]
            diff = more - less
            if abs(diff) < 2:
                out.append(Finding(f"같은 달끼리 견주면 비가 많았던 달({fmt_num(more)})과 적었던 달({fmt_num(less)})의 매출 지수(평소=100) 차이가 {abs(diff):.1f}포인트로 뚜렷하지 않습니다."))
            else:
                word = "늘었" if diff > 0 else "줄었"
                out.append(Finding(f"같은 달끼리 견주면 비가 평소보다 많았던 달의 매출 지수는 {fmt_num(more)}, 적었던 달은 {fmt_num(less)}(평소=100)로 "
                                   f"{abs(diff):.1f}포인트 {word}습니다."))
    elif step_id == "store_trend":
        d = df.dropna(subset=["store_idx"])
        if len(d) >= 2:
            a, b = d.iloc[0], d.iloc[-1]
            out.append(Finding(f"{ctx.get('city_label', who)} {industry} 매장은 {a['ym']} 대비 {b['ym']}에 {b['store_idx'] - 100:+.1f}% 변했습니다(센터 표는 시 단위라 동네별 수는 알 수 없습니다)."))
    elif step_id == "peer_trend":
        d = df.dropna(subset=["sales_idx"])
        if len(d) >= 2:
            out.append(Finding(f"{who}의 {industry} 월 카드매출은 {d.iloc[0]['ym']} 대비 {d.iloc[-1]['ym']}에 {d.iloc[-1]['sales_idx'] - 100:+.1f}% 변했습니다."))
    elif step_id == "real_tour_time":
        d = df.dropna(subset=["sales_share_pct"]).sort_values("sales_share_pct", ascending=False)
        if len(d) >= 2:
            out.append(Finding(f"[실데이터 2015~16년] 관광객 카드 결제는 {d.iloc[0]['time_band']}({TB_HOURS.get(d.iloc[0]['time_band'], '')})에 {_pct(d.iloc[0]['sales_share_pct'])}%, "
                               f"{d.iloc[1]['time_band']}에 {_pct(d.iloc[1]['sales_share_pct'])}%가 몰렸습니다(전 업종 합계라 내 업종과 다를 수 있습니다)."))
    elif step_id == "real_tour_dow":
        d = df.dropna(subset=["sales_share_pct"]).set_index("dow")["sales_share_pct"]
        if len(d) >= 5:
            top = d.sort_values(ascending=False)
            out.append(Finding(f"[실데이터 2015~16년] 내국인 관광객 카드 결제는 {top.index[0]}요일에 {_pct(top.iloc[0])}%로 가장 많았습니다(전 업종 합계)."))
    elif step_id == "real_resident_hourly":
        d = df.dropna(subset=["pop_share_pct"]).sort_values("pop_share_pct", ascending=False)
        if len(d) >= 2:
            n = int(df["lot_cnt"].max()) if "lot_cnt" in df and df["lot_cnt"].notna().any() else 0
            out.append(Finding(f"[실데이터 2015-11] {who} 도민 유동인구는 {d.iloc[0]['time_band']}({TB_HOURS.get(d.iloc[0]['time_band'], '')})에 {_pct(d.iloc[0]['pop_share_pct'])}%로 가장 많습니다"
                               + (f"(지번 표본 {n}곳 기준, 표본이 적은 동네는 참고만 하세요)." if n else ".")))
    elif step_id == "real_cafe_density":
        if emd:
            r, n, v = rank_of(df, "store_cnt", "emd", emd)
            if r is not None:
                share = df[df["emd"] == emd]["store_share_pct"].iloc[0]
                out.append(Finding(f"[실데이터 2023-03] 서귀포시에서 {emd}의 {industry}(휴게음식점 신고 업소)는 {fmt_num(v, 0)}곳으로 {n}개 읍면동 중 {r}위, 서귀포시 전체의 {_pct(share)}%입니다."))
            else:
                out.append(Finding(f"[실데이터 2023-03] {emd}에서 찾은 {industry} 신고 업소가 없습니다(일부 동은 주소로 읍면동을 정할 수 없어 빠집니다)."))
        else:
            top = _top(df, "store_cnt", "emd")
            out.append(Finding(f"[실데이터 2023-03] 서귀포시에서 {industry} 신고 업소가 많은 읍면동은 {', '.join(top)} 순입니다."))
    elif step_id == "real_bus_stops":
        r = df.iloc[0]
        if pd.notna(r.get("stop_cnt")):
            tot = r.get("total_stop_cnt")
            out.append(Finding(f"[실데이터 2022-11] 정류소 이름·주변설명에 {emd}의 이름이 든 버스 정류소는 {fmt_num(r['stop_cnt'], 0)}곳입니다"
                               + (f"(제주 전체 {fmt_num(tot, 0)}곳 중). 이름으로만 찾은 참고치입니다." if pd.notna(tot) else ".")))
    return out
