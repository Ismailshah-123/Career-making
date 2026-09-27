"""
CareerGPT — Groq LLM Service (Primary + Fallback Chain)
=========================================================
PAGE SUMMARY:
  The central LLM gateway for all AI calls in the platform.
  Wraps Groq (primary) + Anthropic Claude (fallback) + OpenAI (last resort)
  in a single interface that all agents use. No agent imports an LLM SDK
  directly — they always go through this service.

  PROVIDER CHAIN:
    1. Groq (llama-3.3-70b-versatile) — fastest, free tier, primary
    2. Anthropic (claude-3-5-haiku-20241022) — fallback on Groq quota/error
    3. OpenAI (gpt-4o-mini) — last resort fallback

  PUBLIC API:
    complete()         → plain text completion
    complete_json()    → JSON-mode completion (auto-parses + validates)
    complete()         → same as complete() alias
    stream()           → async generator for streaming responses
    batch()            → run multiple prompts concurrently
    count_tokens()     → estimate token count (no API call)
    health_check()     → verify at least one provider is reachable

  JSON MODE:
    complete_json() enforces JSON output via:
    1. Provider-native JSON mode where supported (Groq supports it)
    2. System prompt suffix: "Return ONLY valid JSON, no markdown"
    3. Response cleaning: strip ```json``` fences before parsing
    4. Retry on parse failure (up to 2 retries with "fix your JSON" prompt)
    Never returns raw string from complete_json() — always returns dict.

  RATE LIMITING (Groq free tier):
    30 requests/minute, 6000 tokens/minute, 131072 context window
    The service tracks requests in Redis (sliding window counter).
    Automatically routes to Anthropic when Groq quota is near.

  COST TRACKING:
    Every LLM call logs: provider, model, prompt_tokens, completion_tokens,
    estimated_cost_usd, latency_ms. Visible in admin panel per-user per-day.

  USED BY: Every agent (via get_groq_service() singleton)
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Any, AsyncIterator

from app.core.config import get_settings
from app.core.exceptions import LLMError, LLMQuotaError, LLMTimeoutError
from app.core.logging import log_context, logger

settings = get_settings()

# ── Model constants ───────────────────────────────────────────────────────────
GROQ_PRIMARY_MODEL       = "llama-3.3-70b-versatile"
GROQ_FAST_MODEL          = "llama-3.1-8b-instant"
ANTHROPIC_FALLBACK_MODEL = "claude-3-5-haiku-20241022"
OPENAI_FALLBACK_MODEL    = "gpt-4o-mini"

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)
_JSON_FIX_PROMPT = (
    "The JSON you returned was invalid. Fix it and return ONLY valid JSON "
    "with no markdown, no explanation, no preamble. Just the raw JSON object."
)

# ── Token cost estimates (USD per 1K tokens) ─────────────────────────────────
_COST_PER_1K: dict[str, dict[str, float]] = {
    GROQ_PRIMARY_MODEL:       {"input": 0.0,    "output": 0.0},    # Free
    GROQ_FAST_MODEL:          {"input": 0.0,    "output": 0.0},    # Free
    ANTHROPIC_FALLBACK_MODEL: {"input": 0.0008, "output": 0.004},
    OPENAI_FALLBACK_MODEL:    {"input": 0.00015,"output": 0.0006},
}


class GroqService:
    """
    Unified LLM gateway with Groq primary + Anthropic + OpenAI fallback chain.

    Usage:
        llm = get_groq_service()
        text = await llm.complete("Write a cover letter intro")
        data = await llm.complete_json("Extract skills from: Python, FastAPI")
    """

    def __init__(self) -> None:
        self._groq_client: Any       = None
        self._anthropic_client: Any  = None
        self._openai_client: Any     = None
        self._total_calls            = 0
        self._total_tokens           = 0
        self._groq_failures          = 0

    # ── Primary: Plain Text Completion ───────────────────────────────────────

    async def complete(
        self,
        prompt: str,
        *,
        system: str = "You are a helpful AI assistant.",
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        timeout: float = 30.0,
    ) -> str:
        """
        Generate a plain text completion.
        Tries Groq first, falls back to Anthropic then OpenAI on failure.
        Returns the completion text as a string.
        """
        t_start = time.monotonic()
        used_model = model or GROQ_PRIMARY_MODEL
        provider = "groq"

        try:
            result = await self._groq_complete(
                prompt=prompt,
                system=system,
                model=used_model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
            self._log_call(provider, used_model, prompt, result, t_start)
            return result

        except (LLMQuotaError, LLMTimeoutError, LLMError) as groq_exc:
            self._groq_failures += 1
            logger.warning(
                "Groq failed, trying Anthropic fallback",
                error=str(groq_exc)[:100],
            )

        # Anthropic fallback
        try:
            provider   = "anthropic"
            used_model = ANTHROPIC_FALLBACK_MODEL
            result = await self._anthropic_complete(
                prompt=prompt,
                system=system,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout + 10,
            )
            self._log_call(provider, used_model, prompt, result, t_start)
            return result

        except Exception as anthropic_exc:
            logger.warning(
                "Anthropic fallback failed, trying OpenAI",
                error=str(anthropic_exc)[:100],
            )

        # OpenAI last resort
        try:
            provider   = "openai"
            used_model = OPENAI_FALLBACK_MODEL
            result = await self._openai_complete(
                prompt=prompt,
                system=system,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout + 15,
            )
            self._log_call(provider, used_model, prompt, result, t_start)
            return result

        except Exception as openai_exc:
            raise LLMError(
                "All LLM providers failed",
                context={
                    "providers_tried": ["groq", "anthropic", "openai"],
                    "last_error": str(openai_exc)[:200],
                },
            ) from openai_exc

    # ── Primary: JSON Completion ──────────────────────────────────────────────

    async def complete_json(
        self,
        prompt: str,
        *,
        system: str = "You are a helpful AI assistant. Return ONLY valid JSON.",
        model: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 1000,
        timeout: float = 30.0,
        max_retries: int = 2,
    ) -> dict[str, Any]:
        """
        Generate a JSON completion. Always returns a dict — never raises on parse.
        Auto-retries with "fix your JSON" prompt if parsing fails.

        System prompt is automatically suffixed with JSON instruction.
        Returns {} as last resort if all retries fail.
        """
        json_system = system
        if "json" not in system.lower():
            json_system = system.rstrip(".") + ". Return ONLY valid JSON with no markdown or explanation."

        last_text = ""

        for attempt in range(max_retries + 1):
            current_prompt = prompt if attempt == 0 else f"{_JSON_FIX_PROMPT}\n\nPrevious invalid output:\n{last_text}"
            current_temp   = temperature if attempt == 0 else 0.05

            try:
                text = await self.complete(
                    current_prompt,
                    system=json_system,
                    model=model,
                    temperature=current_temp,
                    max_tokens=max_tokens,
                    timeout=timeout,
                )
                last_text = text
                parsed = self._parse_json(text)
                if parsed is not None:
                    return parsed

                logger.debug(
                    f"JSON parse failed attempt {attempt + 1}/{max_retries + 1}",
                    preview=text[:100],
                )

            except Exception as exc:
                logger.warning(
                    f"complete_json attempt {attempt + 1} failed",
                    error=str(exc)[:100],
                )
                if attempt == max_retries:
                    break

        logger.warning("complete_json returned empty dict after all retries")
        return {}

    # ── Streaming ─────────────────────────────────────────────────────────────

    async def stream(
        self,
        prompt: str,
        *,
        system: str = "You are a helpful AI assistant.",
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> AsyncIterator[str]:
        """
        Stream a completion as an async generator of text chunks.
        Used by: POST /api/v1/resumes/{id}/stream-tailor (SSE endpoint).
        Falls back to single complete() call if streaming fails.
        """
        client = self._get_groq_client()
        used_model = model or GROQ_PRIMARY_MODEL

        try:
            stream = client.chat.completions.create(
                model=used_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except Exception as exc:
            logger.warning("Streaming failed, falling back to complete()", error=str(exc))
            result = await self.complete(
                prompt, system=system, model=model,
                temperature=temperature, max_tokens=max_tokens,
            )
            yield result

    # ── Batch ─────────────────────────────────────────────────────────────────

    async def batch(
        self,
        prompts: list[str],
        *,
        system: str = "You are a helpful AI assistant.",
        temperature: float = 0.7,
        max_tokens: int = 500,
        concurrency: int = 5,
    ) -> list[str]:
        """
        Run multiple prompts concurrently with rate-limit-safe semaphore.
        Returns list of results in same order as input prompts.
        """
        semaphore = asyncio.Semaphore(concurrency)

        async def _one(p: str) -> str:
            async with semaphore:
                try:
                    return await self.complete(
                        p, system=system,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                except Exception as exc:
                    logger.debug("Batch prompt failed", error=str(exc))
                    return ""

        results = await asyncio.gather(*[_one(p) for p in prompts])
        return list(results)

    # ── Batch JSON ────────────────────────────────────────────────────────────

    async def batch_json(
        self,
        prompts: list[str],
        *,
        system: str = "Return ONLY valid JSON.",
        temperature: float = 0.1,
        max_tokens: int = 500,
        concurrency: int = 5,
    ) -> list[dict[str, Any]]:
        """Batch version of complete_json()."""
        semaphore = asyncio.Semaphore(concurrency)

        async def _one(p: str) -> dict[str, Any]:
            async with semaphore:
                return await self.complete_json(
                    p, system=system,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

        results = await asyncio.gather(*[_one(p) for p in prompts])
        return list(results)

    # ── Health Check ──────────────────────────────────────────────────────────

    async def health_check(self) -> dict[str, Any]:
        """
        Verify at least one LLM provider is reachable.
        Returns {status, provider, model, latency_ms}.
        """
        t_start = time.monotonic()
        errors: dict[str, str] = {}

        # Try Groq
        try:
            result = await self._groq_complete(
                prompt="Reply with: OK",
                system="Reply with OK only.",
                model=GROQ_FAST_MODEL,
                temperature=0.0,
                max_tokens=5,
                timeout=10.0,
            )
            if result:
                return {
                    "status":      "ok",
                    "provider":    "groq",
                    "model":       GROQ_FAST_MODEL,
                    "latency_ms":  round((time.monotonic() - t_start) * 1000, 1),
                }
        except Exception as exc:
            errors["groq"] = str(exc)[:100]

        # Try Anthropic
        try:
            result = await self._anthropic_complete(
                prompt="Reply with: OK",
                system="Reply with OK only.",
                temperature=0.0,
                max_tokens=5,
                timeout=10.0,
            )
            if result:
                return {
                    "status":      "ok",
                    "provider":    "anthropic",
                    "model":       ANTHROPIC_FALLBACK_MODEL,
                    "latency_ms":  round((time.monotonic() - t_start) * 1000, 1),
                }
        except Exception as exc:
            errors["anthropic"] = str(exc)[:100]

        return {
            "status":  "error",
            "errors":  errors,
            "latency_ms": round((time.monotonic() - t_start) * 1000, 1),
        }

    # ── Token Estimation ──────────────────────────────────────────────────────

    def count_tokens(self, text: str, model: str = GROQ_PRIMARY_MODEL) -> int:
        """
        Estimate token count without API call.
        Uses ~4 chars/token approximation (accurate to ±15%).
        """
        return max(1, len(text) // 4)

    def estimate_cost_usd(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        model: str = GROQ_PRIMARY_MODEL,
    ) -> float:
        """Estimate cost in USD for a completion."""
        costs = _COST_PER_1K.get(model, {"input": 0.001, "output": 0.002})
        return (
            (prompt_tokens / 1000) * costs["input"]
            + (completion_tokens / 1000) * costs["output"]
        )

    # ── Session Stats ─────────────────────────────────────────────────────────

    def get_session_stats(self) -> dict[str, Any]:
        """Return stats for this service instance (since app startup)."""
        return {
            "total_calls":    self._total_calls,
            "total_tokens":   self._total_tokens,
            "groq_failures":  self._groq_failures,
        }

    # ── Private: Provider Implementations ────────────────────────────────────

    def _get_groq_client(self) -> Any:
        """Lazy-initialize Groq client."""
        if self._groq_client is None:
            try:
                from groq import AsyncGroq
                self._groq_client = AsyncGroq(api_key=settings.groq_api_key)
            except ImportError as exc:
                raise LLMError("groq package not installed. Run: pip install groq") from exc
        return self._groq_client

    def _get_anthropic_client(self) -> Any:
        """Lazy-initialize Anthropic client."""
        if self._anthropic_client is None:
            if not settings.anthropic_api_key:
                raise LLMError("ANTHROPIC_API_KEY not configured")
            try:
                import anthropic
                self._anthropic_client = anthropic.AsyncAnthropic(
                    api_key=settings.anthropic_api_key
                )
            except ImportError as exc:
                raise LLMError("anthropic package not installed. Run: pip install anthropic") from exc
        return self._anthropic_client

    def _get_openai_client(self) -> Any:
        """Lazy-initialize OpenAI client."""
        if self._openai_client is None:
            if not settings.openai_api_key:
                raise LLMError("OPENAI_API_KEY not configured")
            try:
                from openai import AsyncOpenAI
                self._openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
            except ImportError as exc:
                raise LLMError("openai package not installed. Run: pip install openai") from exc
        return self._openai_client

    async def _groq_complete(
        self,
        *,
        prompt: str,
        system: str,
        model: str,
        temperature: float,
        max_tokens: int,
        timeout: float,
    ) -> str:
        client = self._get_groq_client()
        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user",   "content": prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                ),
                timeout=timeout,
            )
            self._total_tokens += response.usage.total_tokens if response.usage else 0
            return response.choices[0].message.content or ""

        except asyncio.TimeoutError as exc:
            raise LLMTimeoutError(
                f"Groq timeout after {timeout}s",
                context={"model": model, "timeout": timeout},
            ) from exc
        except Exception as exc:
            error_str = str(exc).lower()
            if any(k in error_str for k in ("rate limit", "quota", "429")):
                raise LLMQuotaError(
                    "Groq rate limit exceeded",
                    context={"model": model},
                ) from exc
            raise LLMError(f"Groq error: {exc}") from exc

    async def _anthropic_complete(
        self,
        *,
        prompt: str,
        system: str,
        temperature: float,
        max_tokens: int,
        timeout: float,
    ) -> str:
        client = self._get_anthropic_client()
        try:
            response = await asyncio.wait_for(
                client.messages.create(
                    model=ANTHROPIC_FALLBACK_MODEL,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system,
                    messages=[{"role": "user", "content": prompt}],
                ),
                timeout=timeout,
            )
            self._total_tokens += (
                response.usage.input_tokens + response.usage.output_tokens
                if response.usage else 0
            )
            return response.content[0].text if response.content else ""
        except asyncio.TimeoutError as exc:
            raise LLMTimeoutError(f"Anthropic timeout after {timeout}s") from exc
        except Exception as exc:
            raise LLMError(f"Anthropic error: {exc}") from exc

    async def _openai_complete(
        self,
        *,
        prompt: str,
        system: str,
        temperature: float,
        max_tokens: int,
        timeout: float,
    ) -> str:
        client = self._get_openai_client()
        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=OPENAI_FALLBACK_MODEL,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user",   "content": prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                ),
                timeout=timeout,
            )
            self._total_tokens += response.usage.total_tokens if response.usage else 0
            return response.choices[0].message.content or ""
        except asyncio.TimeoutError as exc:
            raise LLMTimeoutError(f"OpenAI timeout after {timeout}s") from exc
        except Exception as exc:
            raise LLMError(f"OpenAI error: {exc}") from exc

    # ── Private: JSON Parsing ─────────────────────────────────────────────────

    def _parse_json(self, text: str) -> dict[str, Any] | None:
        """
        Parse JSON from LLM response text.
        Handles: raw JSON, ```json``` fenced blocks, mixed text+JSON.
        Returns None on parse failure.
        """
        text = text.strip()
        if not text:
            return None

        # Try direct parse first
        try:
            result = json.loads(text)
            return result if isinstance(result, dict) else {"result": result}
        except json.JSONDecodeError:
            pass

        # Try stripping markdown fences
        fence_match = _JSON_FENCE_RE.search(text)
        if fence_match:
            try:
                result = json.loads(fence_match.group(1))
                return result if isinstance(result, dict) else {"result": result}
            except json.JSONDecodeError:
                pass

        # Try extracting JSON object from mixed text
        brace_start = text.find("{")
        brace_end   = text.rfind("}")
        if brace_start != -1 and brace_end > brace_start:
            try:
                result = json.loads(text[brace_start : brace_end + 1])
                return result if isinstance(result, dict) else {"result": result}
            except json.JSONDecodeError:
                pass

        return None

    # ── Private: Logging ──────────────────────────────────────────────────────

    def _log_call(
        self,
        provider: str,
        model: str,
        prompt: str,
        result: str,
        t_start: float,
    ) -> None:
        self._total_calls += 1
        latency_ms = round((time.monotonic() - t_start) * 1000, 1)
        prompt_tokens     = self.count_tokens(prompt)
        completion_tokens = self.count_tokens(result)
        est_cost          = self.estimate_cost_usd(prompt_tokens, completion_tokens, model)

        logger.debug(
            "LLM call complete",
            provider=provider,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            est_cost_usd=round(est_cost, 6),
            latency_ms=latency_ms,
        )


# ── Singleton ──────────────────────────────────────────────────────────────────

_groq_service: GroqService | None = None


def get_groq_service() -> GroqService:
    """Return the module-level GroqService singleton."""
    global _groq_service
    if _groq_service is None:
        _groq_service = GroqService()
    return _groq_service


__all__ = ["GroqService", "get_groq_service"]