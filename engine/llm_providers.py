from __future__ import annotations

"""Session-only adapters for the supported LLM providers.

API keys are held in process memory, scoped to a browser-generated session ID,
never written to project metadata, logs or response payloads.  All endpoints
are fixed constants; user input cannot redirect requests to arbitrary hosts.
"""

from dataclasses import dataclass
import json
import os
import re
from threading import RLock
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


PROVIDER_SPECS: dict[str, dict[str, Any]] = {
    "openai": {
        "label": "GPT",
        "default_model": "gpt-5.6-terra",
        "environment_key": "OPENAI_API_KEY",
        "api_style": "Responses API",
        "models": [
            {"id": "gpt-5.6-luna", "label": "Fast / lower cost", "profile": "fast", "tool_use": True},
            {"id": "gpt-5.6-terra", "label": "Balanced", "profile": "balanced", "tool_use": True},
            {"id": "gpt-5.6-sol", "label": "Deep reasoning", "profile": "reasoning", "tool_use": True},
        ],
    },
    "anthropic": {
        "label": "Claude",
        "default_model": "claude-sonnet-5",
        "environment_key": "ANTHROPIC_API_KEY",
        "api_style": "Messages API",
        "models": [
            {"id": "claude-haiku-4-5-20251001", "label": "Fast / lower cost", "profile": "fast", "tool_use": True},
            {"id": "claude-sonnet-5", "label": "Balanced", "profile": "balanced", "tool_use": True},
            {"id": "claude-fable-5", "label": "Deep reasoning", "profile": "reasoning", "tool_use": True},
        ],
    },
    "deepseek": {
        "label": "DeepSeek",
        "default_model": "deepseek-v4-pro",
        "environment_key": "DEEPSEEK_API_KEY",
        "api_style": "Chat Completions API",
        "models": [
            {"id": "deepseek-v4-flash", "label": "Fast", "profile": "fast", "tool_use": True},
            {"id": "deepseek-v4-pro", "label": "Reasoning", "profile": "reasoning", "tool_use": True},
        ],
    },
    "gemini": {
        "label": "Gemini",
        "default_model": "gemini-3.5-flash",
        "environment_key": "GEMINI_API_KEY",
        "api_style": "GenerateContent API",
        "models": [
            {"id": "gemini-3.1-flash-lite", "label": "Fast", "profile": "fast", "tool_use": True},
            {"id": "gemini-3.5-flash", "label": "Balanced", "profile": "balanced", "tool_use": True},
            {"id": "gemini-3.1-pro-preview", "label": "Reasoning", "profile": "reasoning", "tool_use": True},
        ],
    },
}

SESSION_TTL_SECONDS = 12 * 60 * 60
SESSION_PATTERN = re.compile(r"^[A-Za-z0-9_-]{16,128}$")
MODEL_CATALOG_TTL_SECONDS = 10 * 60
TRANSIENT_RETRY_DELAYS_SECONDS = (0.35, 1.0)


@dataclass
class ProviderCredential:
    api_key: str
    model: str
    configured_at: float
    source: str = "browser_session_memory"
    last_success_at: float | None = None
    last_failure_at: float | None = None
    last_error_code: str | None = None
    last_error_message: str | None = None


_SESSION_KEYS: dict[str, dict[str, ProviderCredential]] = {}
_MODEL_CATALOG_CACHE: dict[str, dict[str, dict[str, Any]]] = {}
_LOCK = RLock()


class ProviderRequestError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        code: str = "provider_request_failed",
        http_status: int | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.http_status = http_status


def _is_transient_http_status(status: int) -> bool:
    """Match the bounded retry contract used by the official OpenAI SDK."""
    return status in {408, 409, 429} or status >= 500


def _retry_delay(attempt: int, headers: Any = None) -> float:
    """Use Retry-After when it is short, otherwise deterministic backoff."""
    retry_after = headers.get("Retry-After") if headers is not None else None
    try:
        parsed = float(retry_after)
    except (TypeError, ValueError):
        parsed = 0.0
    if 0.0 < parsed <= 5.0:
        return parsed
    return TRANSIENT_RETRY_DELAYS_SECONDS[min(attempt, len(TRANSIENT_RETRY_DELAYS_SECONDS) - 1)]


def _validate_session_id(session_id: str | None) -> str:
    cleaned = str(session_id or "").strip()
    if not SESSION_PATTERN.fullmatch(cleaned):
        raise ValueError("The assistant session identifier is invalid. Refresh the page and try again.")
    return cleaned


def _prune_sessions(now: float | None = None) -> None:
    current = now or time.time()
    expired = [
        session_id
        for session_id, providers in _SESSION_KEYS.items()
        if not providers or max(item.configured_at for item in providers.values()) < current - SESSION_TTL_SECONDS
    ]
    for session_id in expired:
        _SESSION_KEYS.pop(session_id, None)


def configure_provider(
    *,
    session_id: str,
    provider: str,
    api_key: str | None,
    model: str | None = None,
    clear: bool = False,
) -> dict[str, Any]:
    session_id = _validate_session_id(session_id)
    provider = str(provider).strip().lower()
    if provider not in PROVIDER_SPECS:
        raise ValueError("Choose GPT, Claude, DeepSeek or Gemini.")
    with _LOCK:
        _prune_sessions()
        providers = _SESSION_KEYS.setdefault(session_id, {})
        if clear:
            providers.pop(provider, None)
            _MODEL_CATALOG_CACHE.get(session_id, {}).pop(provider, None)
            if not providers:
                _SESSION_KEYS.pop(session_id, None)
                _MODEL_CATALOG_CACHE.pop(session_id, None)
            return provider_status(session_id)
        cleaned_key = str(api_key or "").strip()
        existing = providers.get(provider)
        if not cleaned_key and existing is not None:
            # A connected provider already has its key in this session. Model
            # changes should update only the selected model, never ask the user
            # to paste the same secret again.
            cleaned_key = existing.api_key
        if not 12 <= len(cleaned_key) <= 500:
            raise ValueError("Enter a complete provider API key. The key is kept only in backend memory for this browser session.")
        cleaned_model = str(model or PROVIDER_SPECS[provider]["default_model"]).strip()
        if not re.fullmatch(r"[A-Za-z0-9._:-]{2,120}", cleaned_model):
            raise ValueError("The model ID contains unsupported characters.")
        providers[provider] = ProviderCredential(cleaned_key, cleaned_model, time.time())
    return provider_status(session_id)


def _environment_credential(provider: str) -> ProviderCredential | None:
    spec = PROVIDER_SPECS[provider]
    api_key = os.getenv(spec["environment_key"], "").strip()
    if not api_key:
        return None
    model = os.getenv(
        f"TWM_{provider.upper()}_MODEL",
        os.getenv(f"PROTOTYPE5_{provider.upper()}_MODEL", spec["default_model"]),
    ).strip() or spec["default_model"]
    return ProviderCredential(api_key, model, time.time(), source="server_environment")


def get_credential(session_id: str | None, provider: str) -> ProviderCredential | None:
    provider = str(provider).strip().lower()
    if provider not in PROVIDER_SPECS:
        return None
    with _LOCK:
        _prune_sessions()
        if session_id and SESSION_PATTERN.fullmatch(str(session_id)):
            credential = _SESSION_KEYS.get(str(session_id), {}).get(provider)
            if credential is not None:
                return credential
    return _environment_credential(provider)


def mark_provider_call_success(session_id: str | None, provider: str) -> None:
    """Record that a real inference request completed for this session/model."""
    if not session_id or not SESSION_PATTERN.fullmatch(str(session_id)):
        return
    with _LOCK:
        credential = _SESSION_KEYS.get(str(session_id), {}).get(str(provider).lower())
        if credential is None:
            return
        credential.last_success_at = time.time()
        credential.last_failure_at = None
        credential.last_error_code = None
        credential.last_error_message = None


def mark_provider_call_failure(
    session_id: str | None,
    provider: str,
    error: Exception,
) -> None:
    """Record a safe failure reason without retaining prompts or provider bodies."""
    if not session_id or not SESSION_PATTERN.fullmatch(str(session_id)):
        return
    with _LOCK:
        credential = _SESSION_KEYS.get(str(session_id), {}).get(str(provider).lower())
        if credential is None:
            return
        credential.last_failure_at = time.time()
        credential.last_error_code = str(getattr(error, "code", "provider_request_failed"))[:80]
        credential.last_error_message = str(error)[:300]


def provider_error_message(error: Exception) -> str:
    """Return the already-sanitised provider failure shown to the user."""
    if isinstance(error, ProviderRequestError):
        return str(error)
    if isinstance(error, TimeoutError):
        return "The provider timed out before returning an answer."
    return "The provider could not complete the inference request."


def provider_status(session_id: str | None = None) -> dict[str, Any]:
    providers = []
    for provider, spec in PROVIDER_SPECS.items():
        credential = get_credential(session_id, provider)
        catalog = _MODEL_CATALOG_CACHE.get(str(session_id or ""), {}).get(provider)
        catalog_is_fresh = bool(catalog and time.time() - float(catalog.get("checked_at", 0)) < MODEL_CATALOG_TTL_SECONDS)
        models = catalog.get("models", []) if catalog_is_fresh else list(spec.get("models", []))
        if not catalog_is_fresh:
            models = [
                {**item, "available": None}
                for item in models
            ]
        catalog_status = "verified" if catalog_is_fresh and catalog.get("status") == "verified" else (
            "unavailable" if catalog_is_fresh and catalog.get("status") == "unavailable" else (
                "not_checked" if credential else "requires_api"
            )
        )
        if credential is None:
            connection = "not_configured"
        elif credential.last_failure_at is not None:
            connection = "inference_failed"
        elif credential.last_success_at is not None:
            connection = "inference_verified"
        else:
            # A key held in memory is not called verified until a real
            # inference request has completed. Model-list access is optional.
            connection = "verification_pending"
        providers.append(
            {
                "id": provider,
                "label": spec["label"],
                "api_style": spec["api_style"],
                "configured": credential is not None,
                "model": credential.model if credential else spec["default_model"],
                "models": models,
                "catalog_status": catalog_status,
                "catalog_checked_at": catalog.get("checked_at") if catalog_is_fresh else None,
                "catalog_error": catalog.get("error") if catalog_is_fresh else None,
                "available_model_count": sum(1 for item in models if item.get("available") is True),
                "custom_model_allowed": False,
                "key_location": credential.source if credential else "not_configured",
                "connection": connection,
                "last_success_at": credential.last_success_at if credential else None,
                "last_failure_at": credential.last_failure_at if credential else None,
                "last_error_code": credential.last_error_code if credential else None,
                "last_error_message": credential.last_error_message if credential else None,
            }
        )
    configured = [item for item in providers if item["configured"]]
    return {
        "status": "llm_ready" if configured else "local_ready",
        "api_configured": bool(configured),
        "configured_provider_count": len(configured),
        "providers": providers,
        "key_persistence": "process_memory_only",
        "session_ttl_hours": SESSION_TTL_SECONDS // 3600,
    }


def _get_json(url: str, headers: dict[str, str], timeout: float = 12.0) -> dict[str, Any]:
    for attempt in range(len(TRANSIENT_RETRY_DELAYS_SECONDS) + 1):
        request = Request(url, method="GET", headers=headers)
        try:
            with urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            if _is_transient_http_status(exc.code) and attempt < len(TRANSIENT_RETRY_DELAYS_SECONDS):
                time.sleep(_retry_delay(attempt, exc.headers))
                continue
            if exc.code == 401:
                message = "The provider rejected this API key. Create or copy an active API secret and try again."
                code = "authentication_invalid"
            elif exc.code == 403:
                message = "This API key cannot read the provider model catalogue."
                code = "model_catalog_permission_denied"
            elif exc.code == 429:
                message = "The provider rate limit was reached while checking available models. Wait briefly and retry."
                code = "rate_limited"
            elif exc.code in {408, 409} or exc.code >= 500:
                message = "The provider model catalogue is temporarily unavailable after bounded retries."
                code = "upstream_unavailable"
            else:
                message = f"The provider returned HTTP {exc.code} while listing models."
                code = "provider_http_error"
            raise ProviderRequestError(message, code=code, http_status=exc.code) from exc
        except (URLError, TimeoutError, ConnectionError) as exc:
            if attempt < len(TRANSIENT_RETRY_DELAYS_SECONDS):
                time.sleep(_retry_delay(attempt))
                continue
            raise ProviderRequestError(
                "The provider model list was unavailable or timed out after bounded retries.",
                code="network_or_timeout",
            ) from exc
        except json.JSONDecodeError as exc:
            raise ProviderRequestError(
                "The provider returned an unreadable model list.",
                code="unreadable_response",
            ) from exc
    raise ProviderRequestError("The provider model list could not be checked.", code="upstream_unavailable")


def _model_is_platform_compatible(provider: str, model_id: str, methods: list[str] | None = None) -> bool:
    """Identify models that can answer text/JSON platform requests.

    Provider model directories also contain embeddings, moderation, image,
    audio and other specialist endpoints. They remain visible in the UI, but
    are marked unavailable for LLM Copilot instead of being silently omitted.
    """
    normalized = str(model_id or "").lower()
    if methods is not None and methods and "generatecontent" not in {str(item).lower() for item in methods}:
        return False
    specialist_tokens = (
        "embedding", "moderation", "dall-e", "image", "vision-only", "whisper",
        "transcri", "speech", "tts", "audio", "realtime", "video", "veo", "sora",
    )
    return not any(token in normalized for token in specialist_tokens)


def _provider_model_entries(provider: str, api_key: str) -> list[dict[str, Any]]:
    """Read a provider's complete advertised model directory."""
    if provider == "openai":
        payload = _get_json(
            "https://api.openai.com/v1/models",
            {"Authorization": f"Bearer {api_key}"},
        )
        return [
            {
                "id": str(item.get("id")),
                "platform_compatible": _model_is_platform_compatible("openai", str(item.get("id"))),
            }
            for item in payload.get("data", [])
            if item.get("id")
        ]
    if provider == "anthropic":
        payload = _get_json(
            "https://api.anthropic.com/v1/models",
            {"x-api-key": api_key, "anthropic-version": "2023-06-01"},
        )
        return [
            {
                "id": str(item.get("id")),
                "platform_compatible": _model_is_platform_compatible("anthropic", str(item.get("id"))),
            }
            for item in payload.get("data", [])
            if item.get("id")
        ]
    if provider == "deepseek":
        payload = _get_json(
            "https://api.deepseek.com/models",
            {"Authorization": f"Bearer {api_key}"},
        )
        return [
            {
                "id": str(item.get("id")),
                "platform_compatible": _model_is_platform_compatible("deepseek", str(item.get("id"))),
            }
            for item in payload.get("data", [])
            if item.get("id")
        ]
    if provider == "gemini":
        payload = _get_json(
            f"https://generativelanguage.googleapis.com/v1beta/models?key={quote(api_key, safe='')}",
            {},
        )
        entries = []
        for item in payload.get("models", []):
            methods = item.get("supportedGenerationMethods", [])
            name = str(item.get("name", ""))
            if name:
                model_id = name.removeprefix("models/")
                entries.append(
                    {
                        "id": model_id,
                        "platform_compatible": _model_is_platform_compatible("gemini", model_id, methods),
                    }
                )
        return entries
    raise ValueError("Choose GPT, Claude, DeepSeek or Gemini.")


def _provider_model_ids(provider: str, api_key: str) -> list[str]:
    """Backward-compatible helper returning the complete model ID list."""
    return [item["id"] for item in _provider_model_entries(provider, api_key)]


def validate_provider_key(*, provider: str, api_key: str) -> None:
    """Verify that a key belongs to the provider selected in the UI."""
    provider = str(provider).strip().lower()
    if provider not in PROVIDER_SPECS:
        raise ValueError("Choose GPT, Claude, DeepSeek or Gemini.")
    cleaned_key = str(api_key or "").strip()
    if not cleaned_key:
        raise ValueError("Enter an API key before connecting this provider.")
    try:
        entries = _provider_model_entries(provider, cleaned_key)
    except ProviderRequestError as exc:
        # A model-directory check is discovery, not authentication. Restricted
        # project keys may allow generation while denying Models, and a
        # temporary catalogue outage says nothing about key validity. Only an
        # explicit 401 is definitive here; real inference is authoritative.
        if exc.code != "authentication_invalid":
            return
        raise ProviderRequestError(
            f"{PROVIDER_SPECS[provider]['label']} connection check failed. {exc}",
            code=exc.code,
            http_status=exc.http_status,
        ) from exc
    if not entries:
        raise ProviderRequestError(
            f"The key was accepted by {PROVIDER_SPECS[provider]['label']}, but no models were returned.",
            code="empty_model_catalog",
            http_status=502,
        )


def refresh_provider_models(*, session_id: str, provider: str) -> dict[str, Any]:
    """Refresh one connected provider's supported-model catalogue."""
    session_id = _validate_session_id(session_id)
    provider = str(provider).strip().lower()
    if provider not in PROVIDER_SPECS:
        raise ValueError("Choose GPT, Claude, DeepSeek or Gemini.")
    credential = get_credential(session_id, provider)
    if credential is None:
        raise ValueError("Connect this provider API before checking its models.")
    spec = PROVIDER_SPECS[provider]
    static_models = {item["id"]: item for item in spec.get("models", [])}
    checked_at = time.time()
    try:
        discovered = {
            item["id"]: item
            for item in _provider_model_entries(provider, credential.api_key)
            if item.get("id")
        }
        discovered_ids = sorted(discovered)
        if not discovered_ids:
            raise ProviderRequestError("Provider returned no models.")
        models = []
        for model_id in discovered_ids:
            base = static_models.get(model_id, {
                "id": model_id,
                "label": "Provider model",
                "profile": "provider",
                "tool_use": True,
            })
            compatible = discovered[model_id].get("platform_compatible", True)
            models.append({
                **base,
                "available": True,
                "platform_compatible": compatible,
                "unavailable_reason": None if compatible else "This model is not a text/JSON platform model.",
            })
        catalog = {"status": "verified", "models": models, "checked_at": checked_at}
    except (ProviderRequestError, ValueError) as exc:
        # Keep the curated catalogue usable when a provider does not expose a
        # listing endpoint. Unknown support is never presented as confirmed.
        catalog = {
            "status": "unavailable",
            "models": [{**item, "available": None} for item in spec.get("models", [])],
            "checked_at": checked_at,
            "error": str(exc),
        }
    with _LOCK:
        _MODEL_CATALOG_CACHE.setdefault(session_id, {})[provider] = catalog
        # A static pre-connection choice may not exist for this account. Once
        # the authoritative directory is known, keep the credential on the
        # first selectable model so the next Copilot request cannot use a
        # fictional or specialist-only fallback ID.
        connected = _SESSION_KEYS.get(session_id, {}).get(provider)
        selectable = [
            item["id"] for item in catalog.get("models", [])
            if item.get("available") is True and item.get("platform_compatible") is not False
        ]
        if connected and selectable and connected.model not in selectable:
            connected.model = selectable[0]
            connected.last_success_at = None
            connected.last_failure_at = None
            connected.last_error_code = None
            connected.last_error_message = None
    return provider_status(session_id)


def _post_json(url: str, body: dict[str, Any], headers: dict[str, str], timeout: float) -> dict[str, Any]:
    encoded_body = json.dumps(body, ensure_ascii=False).encode("utf-8")
    for attempt in range(len(TRANSIENT_RETRY_DELAYS_SECONDS) + 1):
        request = Request(
            url,
            data=encoded_body,
            method="POST",
            headers={"Content-Type": "application/json", **headers},
        )
        try:
            with urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            # Read only the provider's structured error code. Never surface the
            # raw body or provider message because either may echo request text.
            provider_code = ""
            try:
                error_payload = json.loads(exc.read(16_384).decode("utf-8", errors="replace"))
                error_object = error_payload.get("error") if isinstance(error_payload, dict) else None
                if isinstance(error_object, dict):
                    provider_code = str(error_object.get("code") or error_object.get("type") or "").lower()
            except (json.JSONDecodeError, AttributeError, TypeError, ValueError):
                provider_code = ""
            billing_error = provider_code in {
                "insufficient_quota", "billing_hard_limit_reached", "billing_not_active",
            }
            if (
                not billing_error
                and _is_transient_http_status(exc.code)
                and attempt < len(TRANSIENT_RETRY_DELAYS_SECONDS)
            ):
                time.sleep(_retry_delay(attempt, exc.headers))
                continue
            if billing_error:
                message = "The API account has no usable credit or its spending limit has been reached."
                code = "billing_or_quota"
            elif exc.code == 429:
                message = "The provider rate limit was reached after bounded retries. Wait briefly and retry."
                code = "rate_limited"
            elif exc.code in {401, 403}:
                message = "The provider rejected this key or the selected model is not permitted for the account."
                code = "authentication_or_permission"
            elif exc.code == 404:
                message = "The selected model is not available to this API account."
                code = "model_not_available"
            elif exc.code == 400:
                message = "The provider rejected the selected model or request format."
                code = "invalid_provider_request"
            elif exc.code in {408, 409} or exc.code >= 500:
                message = "The provider is temporarily unavailable after bounded retries. Your key has not been classified as invalid."
                code = "upstream_unavailable"
            else:
                message = f"The provider returned HTTP {exc.code} while generating the answer."
                code = provider_code or "provider_http_error"
            raise ProviderRequestError(message, code=code, http_status=exc.code) from exc
        except (URLError, TimeoutError, ConnectionError) as exc:
            if attempt < len(TRANSIENT_RETRY_DELAYS_SECONDS):
                time.sleep(_retry_delay(attempt))
                continue
            raise ProviderRequestError(
                "The provider could not be reached or timed out after bounded retries. Your key has not been classified as invalid.",
                code="network_or_timeout",
            ) from exc
        except json.JSONDecodeError as exc:
            raise ProviderRequestError(
                "The provider returned an unreadable response.", code="unreadable_response"
            ) from exc
    raise ProviderRequestError("The provider request could not be completed.", code="upstream_unavailable")


def _openai_text(payload: dict[str, Any]) -> str:
    if isinstance(payload.get("output_text"), str):
        return payload["output_text"]
    texts: list[str] = []
    for item in payload.get("output", []):
        for content in item.get("content", []) if isinstance(item, dict) else []:
            text = content.get("text") if isinstance(content, dict) else None
            if isinstance(text, str):
                texts.append(text)
    return "\n".join(texts)


def call_provider_json(
    *,
    provider: str,
    credential: ProviderCredential,
    system: str,
    user_payload: dict[str, Any],
    schema: dict[str, Any] | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Call one provider and parse a JSON object from its text response."""

    provider = provider.lower()
    timeout = max(5.0, min(120.0, float(timeout or os.getenv("PROTOTYPE5_LLM_TIMEOUT_S", "45"))))
    user_text = json.dumps(user_payload, ensure_ascii=False)
    json_instruction = "Return one JSON object only, with no markdown fences."
    if schema:
        json_instruction += " It must satisfy this JSON Schema: " + json.dumps(schema, ensure_ascii=False)
    if provider == "openai":
        body: dict[str, Any] = {
            "model": credential.model,
            "input": [
                {"role": "system", "content": [{"type": "input_text", "text": system}]},
                {"role": "user", "content": [{"type": "input_text", "text": json_instruction + "\n" + user_text}]},
            ],
            "store": False,
        }
        if schema:
            body["text"] = {"format": {"type": "json_schema", "name": "prototype5_result", "strict": True, "schema": schema}}
        payload = _post_json(
            "https://api.openai.com/v1/responses",
            body,
            {"Authorization": f"Bearer {credential.api_key}"},
            timeout,
        )
        text = _openai_text(payload)
    elif provider == "anthropic":
        payload = _post_json(
            "https://api.anthropic.com/v1/messages",
            {
                "model": credential.model,
                "max_tokens": 2500,
                "system": system,
                "messages": [{"role": "user", "content": json_instruction + "\n" + user_text}],
            },
            {"x-api-key": credential.api_key, "anthropic-version": "2023-06-01"},
            timeout,
        )
        text = "\n".join(
            item.get("text", "")
            for item in payload.get("content", [])
            if isinstance(item, dict) and item.get("type") == "text"
        )
    elif provider == "deepseek":
        payload = _post_json(
            "https://api.deepseek.com/chat/completions",
            {
                "model": credential.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": json_instruction + "\n" + user_text},
                ],
                "response_format": {"type": "json_object"},
                "stream": False,
            },
            {"Authorization": f"Bearer {credential.api_key}"},
            timeout,
        )
        text = str(payload.get("choices", [{}])[0].get("message", {}).get("content", ""))
    elif provider == "gemini":
        payload = _post_json(
            f"https://generativelanguage.googleapis.com/v1beta/models/{quote(credential.model, safe='.-_')}:generateContent",
            {
                "systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": json_instruction + "\n" + user_text}]}],
                "generationConfig": {"responseMimeType": "application/json"},
            },
            {"x-goog-api-key": credential.api_key},
            timeout,
        )
        text = "\n".join(
            part.get("text", "")
            for candidate in payload.get("candidates", [])
            for part in candidate.get("content", {}).get("parts", [])
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        )
    else:
        raise ValueError("Unsupported LLM provider.")
    if not text.strip():
        raise ProviderRequestError("Provider returned an empty response.")
    try:
        result = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ProviderRequestError("Provider did not return valid JSON.") from exc
    if not isinstance(result, dict):
        raise ProviderRequestError("Provider JSON must be an object.")
    return result


def verify_provider_inference(
    *,
    session_id: str,
    provider: str,
    max_model_attempts: int = 3,
) -> dict[str, Any]:
    """Verify a real inference call and select a working model when possible.

    Model-specific incompatibility may try a small bounded fallback list.
    Billing, authentication, rate-limit and network failures stop immediately
    because changing models cannot resolve them and would waste requests.
    """
    session_id = _validate_session_id(session_id)
    provider = str(provider).strip().lower()
    if provider not in PROVIDER_SPECS:
        raise ValueError("Choose GPT, Claude, DeepSeek or Gemini.")
    credential = get_credential(session_id, provider)
    if credential is None:
        raise ValueError("Verify the provider key before testing inference.")

    catalog = _MODEL_CATALOG_CACHE.get(session_id, {}).get(provider) or {}
    catalog_models = catalog.get("models") or []
    compatible_ids = [
        str(item.get("id"))
        for item in catalog_models
        if item.get("id")
        and item.get("available") is True
        and item.get("platform_compatible") is not False
    ]
    if not compatible_ids:
        compatible_ids = [str(item["id"]) for item in PROVIDER_SPECS[provider].get("models", [])]

    preferred = [
        credential.model,
        PROVIDER_SPECS[provider]["default_model"],
        *[str(item["id"]) for item in PROVIDER_SPECS[provider].get("models", [])],
        *compatible_ids,
    ]
    candidates: list[str] = []
    for model_id in preferred:
        if model_id in compatible_ids and model_id not in candidates:
            candidates.append(model_id)
    candidates = candidates[: max(1, min(3, int(max_model_attempts)))]
    if not candidates:
        raise ValueError("This API did not return a compatible text model for LLM Copilot.")

    original_model = credential.model
    attempts: list[dict[str, str]] = []
    last_error: ProviderRequestError | None = None
    retryable_model_errors = {
        "model_not_available",
        "invalid_provider_request",
        "unreadable_response",
        "empty_response",
        "provider_request_failed",
    }
    test_schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
        "additionalProperties": False,
    }
    for model_id in candidates:
        with _LOCK:
            credential.model = model_id
        try:
            test_result = call_provider_json(
                provider=provider,
                credential=credential,
                system="Return a JSON object with ok set to true. This is a connection test only.",
                user_payload={"request": "verify_inference"},
                schema=test_schema,
                timeout=30.0,
            )
            if test_result.get("ok") is not True:
                raise ProviderRequestError(
                    "The provider returned an invalid connection-test result.",
                    code="invalid_verification_response",
                )
        except ProviderRequestError as exc:
            last_error = exc
            attempts.append({"model": model_id, "status": "failed", "reason": str(exc)})
            if exc.code not in retryable_model_errors:
                break
            continue
        mark_provider_call_success(session_id, provider)
        attempts.append({"model": model_id, "status": "success", "reason": "Inference verified."})
        return {
            **provider_status(session_id),
            "verification": {
                "success": True,
                "provider": provider,
                "selected_model": model_id,
                "attempt_count": len(attempts),
                "attempts": attempts,
                "message": "A real inference request completed successfully.",
            },
        }

    with _LOCK:
        credential.model = original_model
    failure = last_error or ProviderRequestError("The provider did not complete the connection test.")
    mark_provider_call_failure(session_id, provider, failure)
    return {
        **provider_status(session_id),
        "verification": {
            "success": False,
            "provider": provider,
            "selected_model": original_model,
            "attempt_count": len(attempts),
            "attempts": attempts,
            "error_code": failure.code,
            "message": provider_error_message(failure),
        },
    }
