"""Silence chromadb's posthog telemetry. MUST be imported BEFORE chromadb.

Why this exists
---------------
chromadb 0.5.x calls `posthog.capture(distinct_id, event, properties)` (3
positional args) for anonymous usage stats. The posthog version that
lands in our offline wheel set has a different signature, so every event
raises:

    Failed to send telemetry event ClientStartEvent:
        capture() takes 1 positional argument but 3 were given

The errors are non-fatal — ingest and retrieval still work — but they
spam the console.

Per-client `Settings(anonymized_telemetry=False)` is not enough because
chromadb's telemetry singleton is initialised at MODULE LOAD time, with
references captured before any per-client Settings is applied.

The only reliable fix is to neuter the telemetry pathway BEFORE chromadb
is imported anywhere. That requires this module to be the FIRST import
of every entry point.

Usage
-----
At the top of ingest.py / app.py / any other entry point:

    import _silence_chroma  # noqa: F401  — must be the FIRST import
    # ... everything else after this line
"""
import logging
import os

# 1. Tell chromadb (and pydantic-settings) to disable telemetry via env.
#    Both spellings are set because different chromadb versions look in
#    different places.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_TELEMETRY_IMPL", "none")


# 2. Replace posthog.capture (and friends) with a no-op BEFORE chromadb
#    imports posthog. chromadb does `import posthog; posthog.capture(...)`,
#    which looks up the attribute at call time — so a module-level
#    replacement is honoured even after chromadb has already imported the
#    posthog module.
def _noop(*args, **kwargs):
    return None


try:
    import posthog as _posthog
    _posthog.capture = _noop
    _posthog.identify = _noop
    _posthog.alias = _noop
    _posthog.group_identify = _noop
    _posthog.set = _noop
    _posthog.feature_enabled = lambda *a, **k: False
except Exception:
    pass


# 3. If chromadb's telemetry module was already imported by the time we
#    got here (rare but possible if the user re-imports), neuter its
#    Posthog client class too. Belt-and-braces.
try:
    import chromadb.telemetry.product.posthog as _chroma_posthog
    if hasattr(_chroma_posthog, "Posthog"):
        _chroma_posthog.Posthog.capture = lambda self, *a, **k: None
except Exception:
    pass


# 4. Even if a stray posthog call slips through, push the chromadb
#    telemetry logger above ERROR so its "Failed to send telemetry"
#    messages don't reach the console.
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL + 1)
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL + 1)
