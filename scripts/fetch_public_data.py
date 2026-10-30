"""저장소에 넣지 않은 공개 데이터를 원래 공개 주소에서 받아 data/public/ 에 둔다(로그인·키 불필요, 표준 라이브러리만 사용).

저장소에 넣지 않은 이유: 이용조건을 확인하지 못했거나 안내가 서로 어긋나는 파일이기 때문이다(docs/data-usage.md).
받아서 쓰기 전에 아래 이용조건을 직접 확인하세요. 받은 파일은 .gitignore 로 커밋에서 빠집니다.

  python scripts/fetch_public_data.py            # 없는 것만 받기
  python scripts/fetch_public_data.py --check    # 받지 않고 있는지 없는지만 보기
  python scripts/fetch_public_data.py --only bc  # 하나만(cafe 또는 bc)

받는 곳
  cafe  제주특별자치도_서귀포시_휴게음식점현황_20230331.csv  ← 제주데이터허브 데이터 1306 (https://jejudatahub.net/api/data/view?id=1306)
  bc    제주 개별관광(FIT) 소비패턴 BC카드 지역별 분석 CSV 26개 ← 공공데이터포털 15046093 (https://www.data.go.kr/data/15046093/fileData.do)
        (zip 안의 CSV 만 꺼내고 공공누리 마크 그림 파일은 받지 않는다)
앱이 이 파일이 없을 때는 해당 [실데이터] 단계만 "파일 없음 — 받는 법" 안내로 건너뛴다.
"""
from __future__ import annotations

import argparse
import http.cookiejar
import io
import json
import sys
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DEFAULT = ROOT / "data" / "public"
UA = {"User-Agent": "Mozilla/5.0 (jangsa-map fetch_public_data)"}

CAFE_NAME = "제주특별자치도_서귀포시_휴게음식점현황_20230331.csv"
BC_DIR = "제주특별자치도_개별관광(FIT)_증가에_따른_제주_관광객_소비패턴_변화_분석_BC카드_지역별(지리적구분)_20170216"
BC_EXPECT = 26
TIMEOUT_CAP: int | None = None     # 앱(Cloud 데모)이 요청마다 기다리는 최대 초를 줄일 때 설정한다


def _open(opener, url: str, data: dict | None = None, timeout: int = 120) -> bytes:
    if TIMEOUT_CAP:
        timeout = min(timeout, TIMEOUT_CAP)
    body = urllib.parse.urlencode(data).encode() if data is not None else None
    req = urllib.request.Request(url, data=body, headers={**UA, "X-Requested-With": "XMLHttpRequest"} if body else UA)
    with opener.open(req, timeout=timeout) as r:
        return r.read()


def have_cafe(out: Path) -> bool:
    return (out / CAFE_NAME).exists()


def have_bc(out: Path) -> bool:
    d = out / BC_DIR
    return d.is_dir() and len(list(d.glob("*.csv"))) >= BC_EXPECT


def fetch_cafe(out: Path) -> str:
    op = urllib.request.build_opener()
    meta = json.loads(_open(op, "https://jejudatahub.net/api/data/view?id=1306"))
    fi = next(f for f in meta["file"]["fileInfos"] if f["fileName"] == CAFE_NAME)
    raw = _open(op, f"https://jejudatahub.net/api/file?id={fi['id']}&fileId={fi['fileId']}")
    out.mkdir(parents=True, exist_ok=True)
    (out / CAFE_NAME).write_bytes(raw)
    return f"{CAFE_NAME} ({len(raw):,} bytes)"


def fetch_bc(out: Path) -> str:
    op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
    _open(op, "https://www.data.go.kr/data/15046093/fileData.do")          # 세션 쿠키
    info = json.loads(_open(op, "https://www.data.go.kr/tcs/dss/selectFileDataDownload.do",
                            {"publicDataDetailPk": "uddi:241bd332-3e72-409f-b86d-87e22d636d8d", "publicDataPk": "15046093",
                             "atchFileId": "", "fileDetailSn": "1", "publicDataTyCode": "PR0051"}))
    if not info.get("status"):
        raise RuntimeError("공공데이터포털이 다운로드 정보를 주지 않았습니다(페이지가 바뀌었을 수 있음). 위 주소에서 직접 받으세요.")
    atch, sn = info["atchFileId"], info["fileDetailSn"]
    lim = json.loads(_open(op, "https://www.data.go.kr/cmm/cmm/check-limit.json", {"atchFileId": atch, "fileDetailSn": sn}))
    if lim.get("needCaptcha"):
        raise RuntimeError("공공데이터포털이 자동등록 방지 문자를 요구합니다. 브라우저에서 위 주소로 직접 받으세요.")
    raw = _open(op, "https://www.data.go.kr/cmm/cmm/fileDownload.do?" + urllib.parse.urlencode({"atchFileId": atch, "fileDetailSn": sn, "dataNm": "bc"}), timeout=300)
    d = out / BC_DIR
    d.mkdir(parents=True, exist_ok=True)
    n = 0
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        for zi in z.infolist():
            name = zi.filename if zi.flag_bits & 0x800 else zi.filename.encode("cp437").decode("cp949", "replace")
            base = name.rsplit("/", 1)[-1]
            if not base.lower().endswith(".csv"):
                continue                                   # 공공누리 마크 그림 등은 받지 않는다
            (d / base).write_bytes(z.read(zi))
            n += 1
    return f"{BC_DIR}/ 에 CSV {n}개"


JOBS = {"cafe": (have_cafe, fetch_cafe, "서귀포시 휴게음식점 현황(제주데이터허브 1306)"),
        "bc": (have_bc, fetch_bc, "BC카드 관광객 지역별 분석(공공데이터포털 15046093)")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(OUT_DEFAULT), help="저장 폴더(기본 data/public)")
    ap.add_argument("--only", choices=list(JOBS), help="하나만 받기")
    ap.add_argument("--check", action="store_true", help="받지 않고 있는지만 확인")
    ap.add_argument("--force", action="store_true", help="있어도 다시 받기")
    a = ap.parse_args()
    out = Path(a.out)
    rc = 0
    for key, (have, fetch, label) in JOBS.items():
        if a.only and a.only != key:
            continue
        ok = have(out)
        if a.check:
            print(f"[{'있음' if ok else '없음'}] {label}")
            rc = rc or (0 if ok else 1)
            continue
        if ok and not a.force:
            print(f"[건너뜀] {label}: 이미 있음")
            continue
        try:
            print(f"[받는 중] {label} …")
            print("  →", fetch(out))
        except Exception as e:      # 네트워크·페이지 변경: 손으로 받는 길을 알려 준다
            print(f"  ! 실패: {e}", file=sys.stderr)
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
