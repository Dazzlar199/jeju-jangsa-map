"""공개 실데이터(data/public/)를 분석용 표로 정리하는 읽기 코드. 열 이름이 파일마다 달라서 파일별로 따로 쓴다.

모두 제주데이터허브·공공데이터포털에서 로그인 없이 받은 공개 파일이다(이용조건은 docs/data-usage.md).
이 표들은 '실데이터'로 표시하고, 가상 데이터와 섞어 하나의 값으로 계산하지 않는다.
"""
from __future__ import annotations

import glob
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .config import DATA_PUBLIC
from .schema import BC_REGIONS, DOWS, EMD_CITY, TABLES, TIME_BANDS, region_of, to_admin
from .tools.csv_profile import read_csv_auto


@dataclass
class PublicLoad:
    table_id: str
    df: pd.DataFrame
    file_name: str
    period: tuple[str, str] | None
    notes: list[str] = field(default_factory=list)


def _find(pattern: str, folder: Path) -> Path | None:
    hits = sorted(glob.glob(str(folder / "**" / pattern), recursive=True))
    return Path(hits[0]) if hits else None


_CAFE = re.compile(r"커피|카페|coffee|cafe|다방|로스터|티하우스|디저트|브런치|찻집|티룸|스무디|주스|아이스크림|젤라또", re.I)
_BAKERY = re.compile(r"베이커리|제과|빵|bakery|도넛|케이크|떡집|마카롱")
_SNACK = re.compile(r"피자|버거|샌드위치|토스트|핫도그|햄버거")


def classify_license(name: str) -> str | None:
    n = str(name)
    if _CAFE.search(n):
        return "카페"
    if _BAKERY.search(n):
        return "베이커리"
    if _SNACK.search(n):
        return "간이음식"
    return None


_RE_EUPMYEON = re.compile(r"(제주시|서귀포시)\s+([가-힣0-9]+(?:읍|면))")
_RE_DONG = re.compile(r"(제주시|서귀포시)\s+([가-힣0-9]+동)\b")


def emd_from_address(addr: str) -> str | None:
    """주소 글에서 읍면동을 뽑는다. 읍·면이 있으면 그것, 없으면 동 이름(법정동→행정동 매핑)을 쓴다. 못 정하면 None."""
    a = str(addr)
    m = _RE_EUPMYEON.search(a)
    if m and m.group(2) in EMD_CITY:
        return m.group(2)
    m = _RE_DONG.search(a)
    if m:
        e = to_admin(m.group(2))
        if e in EMD_CITY:
            return e
    return None


def load_cafe_license(folder: Path = DATA_PUBLIC) -> PublicLoad | None:
    f = _find(TABLES["cafe_license"].public_glob, folder)
    if not f:
        return None
    df, _ = read_csv_auto(f)
    addr = df["소재지(지번)"].fillna(df.get("소재지(도로명)"))
    emd = addr.map(emd_from_address)
    ind = df["업소명"].map(classify_license)
    total = len(df)
    ok = emd.notna() & ind.notna()
    notes = [f"전체 {total:,}곳 중 읍면동을 알아낸 곳 {int(emd.notna().sum()):,}곳, 카페·베이커리·간이음식으로 분류된 곳 {int(ind.notna().sum()):,}곳, 둘 다 {int(ok.sum()):,}곳만 집계했습니다.",
             "서귀동·보목동 같은 일부 법정동은 행정동(정방·중앙·천지동 등)으로 정할 수 없어 빠졌습니다."]
    g = pd.DataFrame({"emd": emd[ok], "industry": ind[ok]}).groupby(["emd", "industry"], as_index=False).size().rename(columns={"size": "store_cnt"})
    base = str(df["데이터기준일자"].iloc[0])[:7] if "데이터기준일자" in df else "2023-03"
    return PublicLoad("cafe_license", g, f.name, (base, base), notes)


def _read_bc(pattern: str, folder: Path) -> tuple[pd.DataFrame, Path] | None:
    f = _find(pattern, folder)
    if not f:
        return None
    df, _ = read_csv_auto(f)
    return df, f


def load_bc_tour_time(folder: Path = DATA_PUBLIC) -> PublicLoad | None:
    r = _read_bc(TABLES["bc_tour_time"].public_glob, folder)
    if not r:
        return None
    df, f = r
    df = df[df["기간"].astype(str).str.contains("2차년도")]
    long = df.melt(id_vars=["구분"], value_vars=[c for c in TIME_BANDS if c in df.columns], var_name="time_band", value_name="sales_amt")
    long = long.rename(columns={"구분": "region"})
    long["sales_amt"] = pd.to_numeric(long["sales_amt"], errors="coerce")
    return PublicLoad("bc_tour_time", long.dropna().reset_index(drop=True), f.name, ("2015-09", "2016-08"),
                      ["BC카드 가맹점 기준 관광객(개별관광) 결제금액. 2015-09~2016-08 자료라 현재와 다를 수 있습니다."])


def load_bc_tour_dow(folder: Path = DATA_PUBLIC) -> PublicLoad | None:
    r = _read_bc(TABLES["bc_tour_dow"].public_glob, folder)
    if not r:
        return None
    df, f = r
    col = next((c for c in df.columns if "2차년도" in str(c)), None)
    if col is None:
        return None
    out = pd.DataFrame({"region": df["구분"], "dow": df["요일"].astype(str).str.replace("요일", "", regex=False),
                        "sales_amt": pd.to_numeric(df[col], errors="coerce")})
    out = out[out["dow"].isin(DOWS)].dropna()
    return PublicLoad("bc_tour_dow", out.reset_index(drop=True), f.name, ("2015-09", "2016-08"),
                      ["내국인 관광객 요일별 결제금액, 2015-09~2016-08."])


_HOUR_BANDS = {"오전": range(7, 11), "점심": range(11, 14), "오후": range(14, 18), "저녁": range(18, 22),
               "밤": list(range(22, 24)) + list(range(0, 7))}


def load_resident_hourly(folder: Path = DATA_PUBLIC) -> PublicLoad | None:
    f = _find(TABLES["resident_hourly"].public_glob, folder)
    if not f:
        return None
    df, _ = read_csv_auto(f)
    place = df["지번코드 위치"].map(emd_from_address)
    # 열이 두 벌이다: '00시~01시 유동인구'(사람 수)와 '00시~01시 유동인구 비율'(하루 합 1 기준 비율).
    # 비율 열까지 잡으면 같은 시각이 비율 값(0.0006 꼴)으로 덮여 소수 둘째 자리 반올림에서 0이 되므로, 사람 수 열만 쓴다.
    hcols = {int(m.group(1)): c for c in df.columns if (m := re.fullmatch(r"(\d\d)시~\d\d시 유동인구", str(c).strip()))}
    recs = []
    d = df.assign(emd=place).dropna(subset=["emd"])
    for band, hrs in _HOUR_BANDS.items():
        cols = [hcols[h] for h in hrs if h in hcols]
        d[band] = d[cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    g = d.groupby("emd").agg(**{b: (b, "mean") for b in _HOUR_BANDS}, lot_cnt=("emd", "size")).reset_index()
    long = g.melt(id_vars=["emd", "lot_cnt"], value_vars=list(_HOUR_BANDS), var_name="time_band", value_name="pop_avg")
    long["pop_avg"] = long["pop_avg"].round(2)
    notes = [f"지번 표본 {len(df):,}곳 중 읍면동을 알아낸 {len(d):,}곳으로 집계했습니다(읍면동마다 표본 수가 다름: lot_cnt).",
             "제주도민 기준 유동인구(외지인 제외), 2015-11 자료입니다."]
    return PublicLoad("resident_hourly", long, f.name, ("2015-11", "2015-11"), notes)


def load_bus_stops(folder: Path = DATA_PUBLIC) -> PublicLoad | None:
    f = _find(TABLES["bus_stops"].public_glob, folder)
    if not f:
        return None
    df, _ = read_csv_auto(f)
    df = df.dropna(subset=["정류소아이디", "정류소명"])
    out = pd.DataFrame({"stop_id": df["정류소아이디"].astype(str).str.replace(r"\.0$", "", regex=True),
                        "stop_name": df["정류소명"].astype(str), "place_desc": df.get("위치정보(주변설명)", "").astype(str)})
    return PublicLoad("bus_stops", out, f.name, ("2022-11", "2022-11"),
                      ["정류소 ID 는 9자리(공공데이터포털)로, 센터 3-03·3-04 의 7자리 ID 와 같은 체계인지 확인되지 않았습니다. 이름·주변설명 글자로만 찾습니다."])


LOADERS = [load_cafe_license, load_bc_tour_time, load_bc_tour_dow, load_resident_hourly, load_bus_stops]


def load_all(folder: Path | str = DATA_PUBLIC) -> list[PublicLoad]:
    folder = Path(folder)
    out = []
    for fn in LOADERS:
        try:
            r = fn(folder)
        except Exception:      # 파일 형식이 달라도 앱 전체가 멈추지 않게
            r = None
        if r is not None and len(r.df):
            out.append(r)
    return out
