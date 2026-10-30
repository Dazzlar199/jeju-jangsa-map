"""Cloud 데모 전용: 저장소에 없는 공개 파일(휴게음식점·BC카드)을 첫 실행 때 원래 공개 주소에서 받는다.

로직은 scripts/fetch_public_data.py 를 그대로 불러 쓴다(출처·이용조건은 그 파일 맨 위와 docs/data-usage.md).
받기에 실패하면(네트워크·페이지 변경) 예외 없이 결과만 돌려주고, 앱은 지금처럼 "파일 없음" 안내로 건너뛴다.
센터 PC(run.bat)에서는 부르지 않는다(DEMO 일 때만 app.py 가 부름).
"""
from __future__ import annotations

import importlib.util
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .config import DATA_PUBLIC, ROOT

REQUEST_TIMEOUT = 20     # 초. 요청 한 번이 이 시간 넘게 응답이 없으면 그 파일은 포기

# 마지막 받기 결과({키: "ok"|"있음"|"실패: 사유"}). 앱이 "파일 없음"과 "자동 받기 실패(사유)"를 구별해 안내하는 데 쓴다.
# 센터 PC(DEMO 아님)에서는 받기를 하지 않으므로 비어 있다.
STATUS: dict[str, str] = {}


def failure_reason(key: str) -> str | None:
    """그 파일을 자동으로 받다가 실패했으면 사유 글, 아니면 None."""
    v = STATUS.get(key, "")
    return v[len("실패:"):].strip() if v.startswith("실패") else None


def _load_script():
    spec = importlib.util.spec_from_file_location("fetch_public_data", ROOT / "scripts" / "fetch_public_data.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fetch_missing(out: Path | str = DATA_PUBLIC, timeout: int = REQUEST_TIMEOUT, script=None) -> dict[str, str]:
    """없는 파일만 받는다. 돌려주는 값: {키: "ok" | "있음" | "실패: 사유"}. 예외는 밖으로 내지 않는다."""
    out = Path(out)
    try:
        mod = script or _load_script()
        mod.TIMEOUT_CAP = timeout
    except Exception as e:
        res = {k: f"실패: {e}" for k in ("cafe", "bc")}
        STATUS.clear()
        STATUS.update(res)
        return res
    todo = {k: j for k, j in mod.JOBS.items() if not j[0](out)}
    res = {k: "있음" for k in mod.JOBS if k not in todo}

    def one(item):
        k, (_, fetch, _label) = item
        try:
            fetch(out)
            return k, "ok"
        except Exception as e:
            return k, f"실패: {type(e).__name__}: {str(e)[:120]}"

    if todo:
        with ThreadPoolExecutor(max_workers=len(todo)) as ex:
            res.update(dict(ex.map(one, todo.items())))
    STATUS.clear()
    STATUS.update(res)
    return res
