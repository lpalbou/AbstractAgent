"""Provider/model default resolution for the agent factories.

Why this exists (fable5 B-F8, promoted 2026-07-13 on operator green light):
the three factories hardcoded ``ollama`` / ``qwen3:1.7b-q4_K_M``. That pair is
not defensible for what these loops ask of it — ~19 tool schemas plus batching
instructions (plus a structured-output verifier now that review defaults on)
in front of a 1.7B Q4 model, on a server that silently truncates at its
default context window. The ecosystem pattern (gateway, flow) is to resolve
missing provider/model from AbstractCore's configured global defaults and
label any last-resort literal with ``#FALLBACK``.

Resolution order (presence-based; explicit arguments always win):

1. both ``provider`` and ``model`` given -> use them, no warning;
2. both missing -> AbstractCore config ``default_models.global_provider/global_model``
   when BOTH are configured (they are set as a pair by ``abstractcore --config``);
   else the packaged pair with a ``#FALLBACK`` warning naming the remedy;
3. ``provider`` given, ``model`` missing -> the config model only when the
   config provider MATCHES; the packaged model only for the packaged provider
   (an Ollama tag on another provider is a guaranteed wrong shape); any other
   combination REFUSES with a ``ValueError`` naming the fix — shipping a
   wrong-shaped model name to the wire would be the silent-fallback class with
   a worse provider-side error (design adversary P1, 2026-07-13);
4. ``model`` given, ``provider`` missing -> the config provider only when the
   config model MATCHES; otherwise REFUSES (a model tag does not identify its
   server).
"""

from __future__ import annotations

import logging
import warnings
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Packaged last-resort pair. This is deliberately ABSTRACTCORE'S OWN Ollama
# default literal (`OllamaProvider.__init__` default model) so the stack has
# ONE packaged Ollama pair, not two drifting copies — and the instruct-2507
# variant over the bare `qwen3:4b` tag because the bare tag resolves to the
# hybrid-THINKING build (long <think> preambles burning latency and output
# budget in a tool loop), while 2507-instruct is tuned for exactly this
# workload (design adversary find, 2026-07-13). Docs keep an explicit-model
# example regardless.
FALLBACK_PROVIDER = "ollama"
FALLBACK_MODEL = "qwen3:4b-instruct-2507-q4_K_M"


def _configured_global_defaults() -> Tuple[Optional[str], Optional[str]]:
    """AbstractCore's configured global default pair, or (None, None).

    Never raises: an unreadable/absent config means "not configured" — the
    caller falls back loudly. Import stays inside the function so the agents
    package keeps working against cores without the config manager. Cost note:
    ``get_config_manager()`` is a lazy process singleton that AbstractCore's
    own provider constructor already initializes — this adds no new side
    effects at factory time.
    """
    try:
        from abstractcore.config.manager import get_config_manager

        dm = get_config_manager().config.default_models
        provider = str(getattr(dm, "global_provider", "") or "").strip() or None
        model = str(getattr(dm, "global_model", "") or "").strip() or None
        return provider, model
    except Exception:
        return None, None


def _warn(message: str) -> None:
    """Loudness parity: UserWarning for interactive callers (the only channel
    that exists before a run does) + a log record for server hosts that
    collect logging but swallow warnings."""
    warnings.warn(message, UserWarning, stacklevel=4)
    logger.warning(message)


def resolve_provider_model(
    provider: Optional[str],
    model: Optional[str],
) -> Tuple[str, str]:
    """Resolve the effective (provider, model) pair for a factory.

    Explicit values win untouched. Missing halves resolve from AbstractCore
    config global defaults when CONSISTENT; the packaged fallback pair applies
    with a loud ``#FALLBACK`` warning; inconsistent partial specs REFUSE with
    a ``ValueError`` naming the fix (never a wrong-shaped guess).
    """
    p = str(provider or "").strip() or None
    m = str(model or "").strip() or None
    if p and m:
        return p, m

    cfg_provider, cfg_model = _configured_global_defaults()

    if not p and not m:
        if cfg_provider and cfg_model:
            return cfg_provider, cfg_model
        _warn(
            "#FALLBACK no provider/model given and no AbstractCore global default configured — "
            f"using packaged default {FALLBACK_PROVIDER}/{FALLBACK_MODEL}. Configure defaults with "
            "`abstractcore --config` or pass provider=/model= explicitly."
        )
        return FALLBACK_PROVIDER, FALLBACK_MODEL

    if p and not m:
        if cfg_provider == p and cfg_model:
            return p, cfg_model
        if p == FALLBACK_PROVIDER:
            _warn(
                f"#FALLBACK provider '{p}' given without a model — using the packaged default "
                f"'{FALLBACK_MODEL}'. Pass model= explicitly to pin."
            )
            return p, FALLBACK_MODEL
        raise ValueError(
            f"provider '{p}' was given without a model, and the configured AbstractCore global "
            f"default pair ({cfg_provider or 'unset'}/{cfg_model or 'unset'}) is for a different "
            "provider — refusing to guess a model name for a foreign provider (it would fail at "
            "the wire with a worse error). Pass model= explicitly, or set matching defaults via "
            "`abstractcore --config`."
        )

    # model without provider
    assert m is not None
    if cfg_provider and cfg_model == m:
        return cfg_provider, m
    raise ValueError(
        f"model '{m}' was given without a provider, and it does not match the configured "
        f"AbstractCore global default model ({cfg_model or 'unset'}) — a model tag does not "
        "identify its server; refusing to guess. Pass provider= explicitly."
    )
