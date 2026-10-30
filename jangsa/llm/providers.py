"""LLM 공급자. 외부 SDK 없이 표준 라이브러리(urllib)로 REST 를 호출한다(설치 패키지를 늘리지 않기 위해).

- 온도 0 고정, JSON 응답 요청
- 키는 화면 입력값 또는 프로젝트 루트의 .env 파일에서만 읽는다. 시스템 환경변수·다른 폴더의 키 파일은 보지 않는다.
- transport 를 바꿔 끼울 수 있어 네트워크 없이 요청 형식을 시험할 수 있다.
"""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable

from ..config import ROOT, llm_config


class LLMError(Exception):
    pass


@dataclass
class LLMResponse:
    text: str
    in_tokens: int
    out_tokens: int
    model: str
    ms: int
    cost_usd: float


def estimate_tokens(text: str) -> int:
    """대략 추정(한글은 글자당 약 0.7토큰, 영문·숫자는 4자당 1토큰)."""
    ko = len(re.findall(r"[가-힣]", text))
    other = len(text) - ko
    return int(ko * 0.7 + other / 4) + 1


def load_local_env(path=None) -> dict[str, str]:
    """프로젝트 루트의 .env 만 읽는다(없으면 빈 딕셔너리). 시스템 환경변수는 쓰지 않는다."""
    p = path or (ROOT / ".env")
    out: dict[str, str] = {}
    try:
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip('"').strip("'")
    except OSError:
        pass
    return out


Transport = Callable[[str, dict, dict, int], dict]


def _http_post(url: str, headers: dict, body: dict, timeout: int) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), method="POST",
                                 headers={"Content-Type": "application/json", **headers})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:400]
        raise LLMError(f"HTTP {e.code}: {detail}") from e
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise LLMError(f"네트워크 오류: {e}") from e


class Provider:
    name = "base"
    is_offline = False

    def __init__(self, model: str, api_key: str = "", price: dict | None = None, transport: Transport | None = None,
                 timeout: int | None = None, max_tokens: int | None = None):
        cfg = llm_config()
        self.model = model
        self.api_key = api_key
        self.price = price or {"in": 0.0, "out": 0.0}
        self.transport = transport or _http_post
        self.timeout = timeout or cfg["timeout_sec"]
        self.max_tokens = max_tokens or cfg["max_output_tokens"]

    def build_request(self, system: str, user: str) -> tuple[str, dict, dict]:
        raise NotImplementedError

    def parse_response(self, data: dict) -> tuple[str, int, int]:
        raise NotImplementedError

    def complete(self, system: str, user: str) -> LLMResponse:
        if not self.api_key:
            raise LLMError("API 키가 없습니다. 화면에서 키를 입력하거나 .env 를 채우세요.")
        url, headers, body = self.build_request(system, user)
        t0 = time.time()
        data = self.transport(url, headers, body, self.timeout)
        text, tin, tout = self.parse_response(data)
        cost = (tin * self.price["in"] + tout * self.price["out"]) / 1_000_000
        return LLMResponse(text, tin, tout, self.model, int((time.time() - t0) * 1000), cost)


class OpenAIProvider(Provider):
    name = "openai"

    def build_request(self, system, user):
        body = {"model": self.model, "temperature": 0, "max_completion_tokens": self.max_tokens,
                "response_format": {"type": "json_object"},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        return "https://api.openai.com/v1/chat/completions", {"Authorization": f"Bearer {self.api_key}"}, body

    def parse_response(self, data):
        try:
            u = data.get("usage", {})
            return data["choices"][0]["message"]["content"], int(u.get("prompt_tokens", 0)), int(u.get("completion_tokens", 0))
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"OpenAI 응답 형식이 예상과 다릅니다: {str(data)[:200]}") from e

    def complete(self, system, user):
        try:
            return super().complete(system, user)
        except LLMError as e:       # 일부 새 모델은 temperature 를 받지 않는다 → 빼고 한 번 더
            if "temperature" in str(e) and "HTTP 400" in str(e):
                orig = self.build_request

                def no_temp(s, u):
                    url, h, b = orig(s, u)
                    b.pop("temperature", None)
                    return url, h, b
                self.build_request = no_temp  # type: ignore[method-assign]
                return super().complete(system, user)
            raise


class AnthropicProvider(Provider):
    name = "anthropic"

    def build_request(self, system, user):
        body = {"model": self.model, "max_tokens": self.max_tokens, "temperature": 0, "system": system,
                "messages": [{"role": "user", "content": user}]}
        return ("https://api.anthropic.com/v1/messages",
                {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"}, body)

    def parse_response(self, data):
        try:
            text = "".join(b.get("text", "") for b in data["content"] if b.get("type") == "text")
            u = data.get("usage", {})
            return text, int(u.get("input_tokens", 0)), int(u.get("output_tokens", 0))
        except (KeyError, TypeError) as e:
            raise LLMError(f"Claude 응답 형식이 예상과 다릅니다: {str(data)[:200]}") from e


class GeminiProvider(Provider):
    name = "gemini"

    def build_request(self, system, user):
        body = {"systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                "generationConfig": {"temperature": 0, "maxOutputTokens": self.max_tokens,
                                     "responseMimeType": "application/json"}}
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        return url, {"x-goog-api-key": self.api_key}, body

    def parse_response(self, data):
        try:
            text = "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"])
            u = data.get("usageMetadata", {})
            return text, int(u.get("promptTokenCount", 0)), int(u.get("candidatesTokenCount", 0))
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"Gemini 응답 형식이 예상과 다릅니다: {str(data)[:200]}") from e


class OfflineProvider(Provider):
    """키 없이 끝까지 도는 기본 모드. 규칙 기반 계획 + 템플릿 해설. 아무것도 외부로 보내지 않는다."""
    name = "offline"
    is_offline = True

    def __init__(self, *a, **k):
        super().__init__("rule-based", "")

    def complete(self, system, user):
        raise LLMError("오프라인 모드에서는 LLM 을 부르지 않습니다.")


_CLASSES = {"openai": OpenAIProvider, "anthropic": AnthropicProvider, "gemini": GeminiProvider}


def make_provider(name: str = "offline", model: str | None = None, api_key: str | None = None,
                  transport: Transport | None = None) -> Provider:
    if name == "offline":
        return OfflineProvider()
    cfg = llm_config()["providers"].get(name)
    if cfg is None or name not in _CLASSES:
        raise LLMError(f"알 수 없는 공급자: {name}")
    key = api_key or load_local_env().get(cfg["key_env"], "")
    return _CLASSES[name](model or cfg["model"], key, cfg["price"], transport)
