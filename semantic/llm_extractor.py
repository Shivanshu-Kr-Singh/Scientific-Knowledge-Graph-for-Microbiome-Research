"""
semantic/llm_extractor.py — LLM-based biomedical entity and relation extractor.

Routes to Ollama (primary) or Gemini (fallback / direct) based on BACKEND_CONFIG.
Uses _JsonFileCache for atomic, persistent caching of extraction results.

Requirements: 3.1, 3.3, 3.4, 3.5, 4.1–4.8, 7.1–7.4, 8.1, 8.3, 8.5, 8.7, 8.8,
              11.1, 11.3, 11.5, 14.1, 14.3, 14.5
"""

import hashlib
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from config import BACKEND_CONFIG
from semantic._cache import _JsonFileCache
from semantic.candidate_store import CandidateEntity, CandidateRelation
from semantic.ollama_client import OllamaClient, OllamaTimeoutError, OllamaUnavailableError

log = logging.getLogger(__name__)

# ─── Cache ────────────────────────────────────────────────────────────────────

_CACHE_PATH = Path(__file__).parent / "cache" / "llm_extract_cache.json"
_cache = _JsonFileCache(_CACHE_PATH)

# ─── Lazy Gemini client ───────────────────────────────────────────────────────


def _get_gemini_client():
    """
    Lazily import and instantiate the Gemini client.
    Only called when the Gemini path is actually taken, so google-genai is not
    required when LLM_BACKEND=ollama.
    """
    try:
        from google import genai  # noqa: PLC0415
        return genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    except ImportError:
        raise ImportError(
            "google-genai is required for Gemini backend. pip install google-genai"
        )


# ─── Extraction prompt ────────────────────────────────────────────────────────

def _build_prompt(text: str) -> str:
    """
    Build the extraction prompt. Input text is truncated to 3,000 characters
    for speed (abstracts are typically 200-400 words).
    /no_think disables qwen3 chain-of-thought for faster responses.
    """
    truncated = text[:3000]
    return (
        "/no_think\n"
        "You are a biomedical extraction engine. Return ONLY a JSON object.\n"
        "No text before or after the JSON. No markdown code fences. No thinking.\n"
        "\n"
        "Required format:\n"
        '{"entities": [{"name": "Lactobacillus", "type": "taxon", "confidence": 0.9, "novel": false}],\n'
        ' "relations": [{"subject": "Lactobacillus", "predicate": "modulates", "object": "gut barrier", "confidence": 0.85}],\n'
        ' "evidence": {"sample_size": 42, "study_design": "RCT", "population": "adult humans"}}\n'
        "\n"
        "TEXT:\n"
        f"{truncated}"
    )


# ─── JSON parsing helpers ─────────────────────────────────────────────────────

def _strip_markdown_fence(raw: str) -> str:
    """
    Strip markdown code fences and qwen3 <think> blocks if present.
    Returns the stripped string (may still be invalid JSON).
    """
    import re as _re
    stripped = raw.strip()

    # Strip qwen3 <think>...</think> blocks
    stripped = _re.sub(r'<think>.*?</think>', '', stripped, flags=_re.DOTALL).strip()

    if stripped.startswith("```"):
        log.warning(
            "LLMExtractor: response begins with a markdown code fence — stripping fences"
        )
        lines = stripped.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines)
    return stripped


def _parse_response(raw: str) -> dict | None:
    """
    Parse a raw LLM response string into a dict.

    - Strips markdown fences if present (Req 3.3).
    - Handles qwen3 trailing newline padding.
    - Returns None and logs ERROR on JSON parse failure (Req 3.1).
    """
    cleaned = _strip_markdown_fence(raw)

    # qwen3 and some models append trailing \n\n\n... padding after the JSON.
    # Find the last closing brace and truncate there.
    last_brace = cleaned.rfind("}")
    if last_brace != -1:
        cleaned = cleaned[: last_brace + 1]

    # Also handle truncated JSON — find the first complete JSON object
    # by trying to parse progressively shorter strings if needed
    try:
        data = json.loads(cleaned)
        if not isinstance(data, dict):
            log.error("LLMExtractor: parsed JSON is not a dict. Raw response: %r", raw[:200])
            return None
        return data
    except json.JSONDecodeError:
        # Try to extract just the entities array if full parse fails
        import re as _re
        entities_match = _re.search(r'"entities"\s*:\s*(\[.*?\])', cleaned, _re.DOTALL)
        if entities_match:
            try:
                entities_list = json.loads(entities_match.group(1))
                return {"entities": entities_list, "relations": [], "evidence": {}}
            except json.JSONDecodeError:
                pass
        log.error("LLMExtractor: failed to parse JSON response. Raw response: %r", raw[:200])
        return None


def _build_result(data: dict) -> tuple[list[CandidateEntity], list[CandidateRelation]]:
    """
    Convert a parsed extraction schema dict into CandidateEntity / CandidateRelation lists.
    Applies safe defaults for missing or wrong-typed fields (Req 3.4, 3.5).
    """
    entities_raw = data.get("entities", [])
    if not isinstance(entities_raw, list):
        entities_raw = []

    relations_raw = data.get("relations", [])
    if not isinstance(relations_raw, list):
        relations_raw = []

    entities = [
        CandidateEntity(
            name=x.get("name", "") if isinstance(x.get("name"), str) else "",
            entity_type=x.get("type", "unknown") if isinstance(x.get("type"), str) else "unknown",
        )
        for x in entities_raw
        if isinstance(x, dict)
    ]

    relations = [
        CandidateRelation(
            subject=x.get("subject", "") if isinstance(x.get("subject"), str) else "",
            predicate=x.get("predicate", "") if isinstance(x.get("predicate"), str) else "",
            object=x.get("object", "") if isinstance(x.get("object"), str) else "",
            confidence=(
                x.get("confidence", 0.8)
                if isinstance(x.get("confidence"), (int, float))
                else 0.8
            ),
        )
        for x in relations_raw
        if isinstance(x, dict)
    ]

    return entities, relations


# ─── Gemini extraction helper ─────────────────────────────────────────────────

def _call_gemini(prompt: str) -> tuple[list[CandidateEntity], list[CandidateRelation]]:
    """
    Invoke the Gemini backend and return parsed results.
    On any exception: log ERROR, return ([], []) (Req 4.8, 8.5).
    """
    try:
        gemini_client = _get_gemini_client()
        model = BACKEND_CONFIG.gemini_extraction_model
        resp = gemini_client.models.generate_content(model=model, contents=prompt)
        raw = resp.text or "{}"
        data = _parse_response(raw)
        if data is None:
            return [], []
        return _build_result(data)
    except Exception as exc:  # noqa: BLE001
        log.error(
            "LLMExtractor: Gemini backend raised %s: %s",
            type(exc).__name__,
            exc,
        )
        return [], []


# ─── LLMExtractor ─────────────────────────────────────────────────────────────


class LLMExtractor:
    """
    Extracts biomedical entities and relations from text using an LLM backend.

    Routes to Ollama or Gemini based on BACKEND_CONFIG.llm_backend.
    Results are cached in semantic/cache/llm_extract_cache.json.
    """

    def extract(
        self, text: str
    ) -> tuple[list[CandidateEntity], list[CandidateRelation]]:
        """
        Extract entities and relations from *text*.

        Decision tree (Req 4.1–4.8):
          1. Empty / whitespace / None → return ([], [])
          2. Cache hit → return cached result
          3. LLM_BACKEND == "gemini" → call Gemini directly
          4. LLM_BACKEND == "ollama" → call OllamaClient
             - On success: parse JSON, write cache, return result
             - On Ollama error + fallback=True: log WARNING, call Gemini
             - On Ollama error + fallback=False: log ERROR, return ([], [])
          5. JSON parse failure: log ERROR, return ([], []), do NOT write cache
        """
        # ── 1. Guard empty / whitespace / None input (Req 4.2) ───────────────
        if not text or not text.strip():
            return [], []

        # ── 2. Cache lookup (Req 4.3, 7.1, 7.2) ─────────────────────────────
        key = hashlib.md5(text.encode("utf-8")).hexdigest()
        cache = _cache.load()
        if key in cache:
            cached_data = cache[key]
            if isinstance(cached_data, dict):
                return _build_result(cached_data)

        # ── 3. Build prompt (Req 4.7, 11.1, 11.3, 11.5) ─────────────────────
        prompt = _build_prompt(text)

        # ── 4. Route to backend ───────────────────────────────────────────────
        if BACKEND_CONFIG.llm_backend == "gemini":
            # Direct Gemini path (Req 8.7, 8.8, 14.3)
            entities, relations = _call_gemini(prompt)
            if entities or relations:
                # Write to cache on success
                data = {
                    "entities": [
                        {"name": e.name, "type": e.entity_type, "confidence": 0.8, "novel": False}
                        for e in entities
                    ],
                    "relations": [
                        {
                            "subject": r.subject,
                            "predicate": r.predicate,
                            "object": r.object,
                            "confidence": r.confidence,
                        }
                        for r in relations
                    ],
                    "evidence": {},
                }
                cache[key] = data
                _cache.save(cache)
            return entities, relations

        # ── Ollama path (Req 4.4, 4.5, 4.6) ──────────────────────────────────
        ollama_client = OllamaClient(BACKEND_CONFIG)
        model = BACKEND_CONFIG.ollama_extraction_model

        try:
            raw = ollama_client.generate(model, prompt)
        except (OllamaUnavailableError, OllamaTimeoutError) as exc:
            if BACKEND_CONFIG.ollama_fallback_to_gemini:
                # Req 4.5, 8.1: log WARNING and activate Gemini fallback
                log.warning(
                    "LLMExtractor: %s — activating Gemini fallback",
                    type(exc).__name__,
                )
                return _call_gemini(prompt)
            else:
                # Req 4.6: log ERROR, return ([], [])
                log.error(
                    "LLMExtractor: %s: %s — returning empty result",
                    type(exc).__name__,
                    exc,
                )
                return [], []

        # ── Parse JSON response ───────────────────────────────────────────────
        data = _parse_response(raw)
        if data is None:
            # Req 3.1, 4.4: parse failure → log ERROR (already done), do NOT cache
            return [], []

        # ── Write to cache atomically (Req 4.4, 7.4) ─────────────────────────
        cache[key] = data
        _cache.save(cache)

        return _build_result(data)
