"""
graph/llm_triple_extractor.py
------------------------------
LLM-based open-world scientific triple extractor.

Extracts (subject, predicate, object) triples from paper sections using Ollama.
Unlike the regex-based SemanticRelationshipExtractor which uses 3 hardcoded
templates, this extractor discovers any scientific relationship expressed in
the text.

Triples are normalized via PredicateRegistry before being added to the graph.
Novel predicates are stored as RELATES_TO edges with the raw predicate preserved.

Only active when the USE_LLM environment variable is set to "true".
"""

import json
import hashlib
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone
from loguru import logger

from graph.predicate_registry import PredicateRegistry

CACHE_PATH = Path(__file__).parent / "triple_cache.json"

TRIPLE_CACHE_VERSION = "v2"  # Bump this when the extraction prompt changes

TRIPLE_EXTRACTION_PROMPT = """/no_think
Extract biomedical (subject, predicate, object) triples from the TEXT below.
Return ONLY valid JSON. No markdown, no explanation.

Rules:
- Only extract relationships explicitly stated in the text
- Subject and object must be specific named entities
- Confidence: 0.9=stated with stats, 0.7=clearly stated, 0.5=implied
- evidence: exact sentence from text supporting this triple

Output format:
{"triples": [{"subject": "...", "subject_type": "taxon|disease|gene|metabolite|treatment|method", "predicate": "...", "object": "...", "object_type": "taxon|disease|gene|metabolite|treatment|method", "confidence": 0.0, "evidence": "..."}]}

TEXT:
{text}"""


class LLMTripleExtractor:
    """
    Extracts open-world scientific triples from paper sections using Ollama.

    Only active when USE_LLM=true is set in the environment.
    Results are cached to avoid redundant API calls.
    """

    def __init__(self) -> None:
        self.predicate_registry = PredicateRegistry()
        self._cache: Dict[str, Any] = self._load_cache()
        self._ollama_client = None
        self._model: str = ""
        self._available = False
        self._setup()

    def _setup(self) -> None:
        # Honour the USE_LLM_LAYER3 gate (falls back to USE_LLM for compatibility)
        layer3_flag = os.getenv("USE_LLM_LAYER3", os.getenv("USE_LLM", "false"))
        if layer3_flag.lower() != "true":
            logger.debug("[LLMTripleExtractor] USE_LLM_LAYER3 != true — triple extraction disabled")
            return

        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from config import BACKEND_CONFIG
            from semantic.ollama_client import OllamaClient
            self._ollama_client = OllamaClient(BACKEND_CONFIG)
            self._model = BACKEND_CONFIG.ollama_extraction_model
            self._available = True
            logger.info("[LLMTripleExtractor] Ollama client ready | model={}", self._model)
        except Exception as exc:
            logger.warning(
                "[LLMTripleExtractor] Ollama unavailable — triple extraction disabled: {}", exc
            )

    def extract_triples(
        self,
        text: str,
        paper_id: str,
        section_type: str = "unknown",
    ) -> List[Dict[str, Any]]:
        """
        Extract (subject, predicate, object) triples from text.

        Returns list of dicts with keys:
          subject, subject_type, predicate, canonical_predicate, predicate_category,
          object, object_type, confidence, evidence, paper_id, section_type,
          is_novel_predicate

        Returns an empty list when LLM is unavailable, disabled via USE_LLM,
        or the text is empty.
        """
        if not self._available or not text or not text.strip():
            return []

        # Check cache
        cache_key = hashlib.md5(
            f"{TRIPLE_CACHE_VERSION}:{paper_id}:{section_type}:{text[:500]}".encode()
        ).hexdigest()
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Build prompt — keep text short so small local models can respond in time
        truncated = text[:2500]
        prompt = TRIPLE_EXTRACTION_PROMPT.replace("{text}", truncated)

        try:
            raw = self._ollama_client.generate(self._model, prompt)
            triples = self._parse_triples(raw, paper_id, section_type)
            # Cache result
            self._cache[cache_key] = triples
            self._save_cache()
            if triples:
                logger.debug(
                    "[LLMTripleExtractor] Extracted {} triples from {} section of {}",
                    len(triples),
                    section_type,
                    paper_id[:30],
                )
            return triples
        except Exception as exc:
            logger.warning(
                "[LLMTripleExtractor] Extraction failed for {}: {}", paper_id[:30], exc
            )
            return []

    def _parse_triples(
        self, raw, paper_id: str, section_type: str
    ) -> List[Dict[str, Any]]:
        """Parse LLM response into normalized triple dicts."""
        # Guard against None, list, dict or any non-string response from LLM
        if raw is None:
            return []
        if not isinstance(raw, str):
            try:
                raw = str(raw)
            except Exception:
                return []
        if not raw.strip():
            return []
        # Strip think tags and markdown fences
        clean = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        if clean.startswith("```"):
            lines = clean.splitlines()[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            clean = "\n".join(lines)

        # Truncate to last closing brace to handle trailing padding
        last_brace = clean.rfind("}")
        if last_brace != -1:
            clean = clean[: last_brace + 1]

        try:
            data = json.loads(clean)
        except json.JSONDecodeError:
            # Try to salvage the triples array
            match = re.search(r'"triples"\s*:\s*(\[.*?\])', clean, re.DOTALL)
            if match:
                try:
                    data = {"triples": json.loads(match.group(1))}
                except Exception:
                    return []
            else:
                return []

        raw_triples = data.get("triples", [])
        if not isinstance(raw_triples, list):
            return []

        results: List[Dict[str, Any]] = []
        for t in raw_triples:
            if not isinstance(t, dict):
                continue
            subject = t.get("subject", "").strip()
            predicate = t.get("predicate", "").strip()
            object_ = t.get("object", "").strip()

            # Basic quality gates
            if not subject or not predicate or not object_:
                continue
            if len(subject) < 2 or len(object_) < 2:
                continue

            try:
                confidence = float(t.get("confidence", 0.7))
            except (TypeError, ValueError):
                confidence = 0.7

            if confidence < 0.5:
                continue

            # Normalize predicate via registry
            canonical_predicate, is_known = self.predicate_registry.normalize(predicate)
            category = self.predicate_registry.get_category(canonical_predicate)

            results.append({
                "subject": subject,
                "subject_type": t.get("subject_type", "unknown"),
                "predicate": predicate,                      # raw predicate
                "canonical_predicate": canonical_predicate,  # normalized
                "predicate_category": category,
                "is_novel_predicate": not is_known,
                "object": object_,
                "object_type": t.get("object_type", "unknown"),
                "confidence": min(1.0, max(0.5, confidence)),
                "evidence": str(t.get("evidence", ""))[:300],
                "paper_id": paper_id,
                "section_type": section_type,
                "extracted_at": datetime.now(timezone.utc).isoformat(),
            })

        return results

    def _load_cache(self) -> Dict:
        if CACHE_PATH.exists():
            try:
                with open(CACHE_PATH) as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_cache(self) -> None:
        try:
            CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump(self._cache, f)
        except Exception as exc:
            logger.warning("[LLMTripleExtractor] Could not save cache: {}", exc)
