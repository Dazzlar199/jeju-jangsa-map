"""규칙 기반 질문 해석(오프라인 모드의 계획 에이전트). 키워드로 의도·읍면동·업종을 뽑는다."""
from __future__ import annotations

import re
from dataclasses import dataclass

from ..schema import EMD_LIST, INDUSTRY_WORDS, UNSUPPORTED_INDUSTRY_WORDS

INTENTS = {
    "location": "입지 진단(여기서 열면 어떤가)",
    "competition": "업종 경쟁(매장이 너무 많은가)",
    "time_pattern": "시간대·요일 패턴",
    "season_weather": "계절·날씨 영향",
    "benchmark": "내 가게 벤치마크(동네 평균과 비교)",
}
_KEYWORDS = [
    ("benchmark", ["우리 가게", "내 가게", "제 가게", "우리가게", "내가게", "동네 평균", "평균보다", "내 매출", "우리 매출", "우리 집 매출", "백분위", "내 위치"]),
    ("competition", ["경쟁", "너무 많", "포화", "매장 수", "매장수", "몇 개", "과밀", "점포 수", "가게가 많", "많은가", "많아"]),
    ("season_weather", ["비 오", "비오", "날씨", "계절", "여름", "겨울", "비가", "기온", "성수기", "비수기"]),
    ("time_pattern", ["시간대", "몇 시", "요일", "주말", "평일", "언제", "저녁", "점심", "아침", "오전", "오후", "새벽", "밤"]),
    ("location", ["내면", "낼까", "열면", "창업", "차리", "개업", "오픈", "입점", "할까", "좋은 동네", "어디", "어때", "입지", "내려", "하면"]),
]


@dataclass
class Parsed:
    intent: str
    emd: str | None
    industry: str | None
    city: str | None
    matched: list[str]
    unsupported: str | None = None


def find_emd(text: str) -> str | None:
    t = re.sub(r"\s+", "", text)
    best, best_len = None, 0
    for e in EMD_LIST:
        if e in t and len(e) > best_len:
            best, best_len = e, len(e)
    if best:
        return best
    for e in EMD_LIST:          # 읍·면·동 을 빼고 말한 경우: "애월에", "성산에"
        stem = re.sub(r"[읍면동]$", "", e)
        if len(stem) >= 2 and not stem[-1].isdigit() and stem in t and len(stem) > best_len:
            best, best_len = e, len(stem)
    return best


def find_industry(text: str) -> str | None:
    t = re.sub(r"\s+", "", text)
    best, best_len = None, 0
    for canon, words in INDUSTRY_WORDS.items():
        for w in [canon] + list(words):
            if w in t and len(w) > best_len:
                best, best_len = canon, len(w)
    return best


def find_unsupported_industry(text: str) -> str | None:
    t = re.sub(r"\s+", "", text)
    return next((w for w in UNSUPPORTED_INDUSTRY_WORDS if w in t), None)


def find_city(text: str) -> str | None:
    t = re.sub(r"\s+", "", text)
    for c in ("서귀포", "제주시"):
        if c in t:
            return "서귀포시" if c == "서귀포" else "제주시"
    return None


def find_intent(text: str) -> tuple[str, list[str]]:
    t = text.replace(" ", "")
    for intent, kws in _KEYWORDS:
        hit = [k for k in kws if k.replace(" ", "") in t]
        if hit:
            return intent, hit
    return "location", []


def parse_question(q: str) -> Parsed:
    intent, hit = find_intent(q)
    emd = find_emd(q)
    ind = find_industry(q)
    return Parsed(intent, emd, ind, None if emd else find_city(q), hit, None if ind else find_unsupported_industry(q))


def missing_inputs(intent: str, emd: str | None, industry: str | None) -> list[str]:
    need = []
    if industry is None:
        need.append("industry")
    if emd is None and intent in ("benchmark", "competition", "time_pattern", "season_weather"):
        need.append("emd")
    return need
