"""
Stage 4 — LLM verifier for borderline papers only.

SUPPORTED PROVIDERS:
ollama    → Local Ollama instance (default, no API key required)

CACHING:
Every LLM response is cached by content hash in data/processed/llm_cache.json.
Re-runs never call the API twice for the same paper — critical for:
- Cost control (local inference, no quota)
- Reproducibility (same verdict every time — important for research)
"""

import os
import json
import hashlib
import requests
from typing import Optional
from pathlib import Path
from dataclasses import dataclass
from loguru import logger

# ── Config ──
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_VERIFIER_MODEL", os.getenv("OLLAMA_EXTRACTION_MODEL", "llama3"))
OLLAMA_TIMEOUT  = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "60"))

CACHE_PATH = Path(__file__).parent.parent / "data" / "processed" / "llm_cache.json"


@dataclass
class LLMVerdict:
    keep:        bool
    confidence:  float
    reason:      str
    cached:      bool = False
    cost_tokens: int  = 0


# ── Prompt ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a biomedical literature relevance classifier for an academic research system.

PROJECT: Literature Mining and Metagenomic Data Preprocessing for HUMAN MICROBIOME STUDIES (2024–2026)

TASK: Determine whether a paper should be INCLUDED for downstream NLP, data extraction, and knowledge graph construction.

INCLUSION CRITERIA — KEEP if paper studies:
- Human subjects (patients, cohorts, volunteers, clinical/RCT/observational studies)
- Microbiome / microbiota / microbial community / gut flora
- Sequencing methods: 16S, shotgun metagenomics, metatranscriptomics, amplicon sequencing
- Body sites: gut, oral, skin, vaginal, lung, nasal, blood, breast milk

REJECT if primary focus is:
- Animal models (mouse, rat, zebrafish, etc.) — even with human microbiota transplant
- Environmental microbiome (soil, marine, wastewater, sediment)
- Food/fermentation (kombucha, kefir, cheese, wine, beer)
- Plant/agricultural microbiome
- Pure pathogen/infectious disease with no microbiome profiling

CONFIDENCE CALIBRATION:
0.90–1.00: Explicit human microbiome + sequencing/metagenomics
0.75–0.89: Strong human microbiome evidence
0.50–0.74: Borderline → REVIEW
0.00–0.49: Reject

OUTPUT FORMAT — Return JSON only, no markdown:
{"keep": true/false, "confidence": 0.0-1.0, "reason": "short explanation", "human": true/false, "microbiome": true/false, "metagenomics": true/false, "animal": true/false, "environmental": true/false, "article_type": "review/cohort/RCT/preprint/etc"}"""

FEW_SHOT_EXAMPLES = [
    {"title": "Decoding the diet-gut-liver axis",
     "output": {"keep": True, "confidence": 0.95, "reason": "human gut microbiome review"}},
    {"title": "Gut microbiome in IBD patients: 16S rRNA cohort study",
     "output": {"keep": True, "confidence": 0.97, "reason": "human IBD cohort with 16S sequencing"}},
    {"title": "Zebrafish gut microbiome response to antibiotic treatment",
     "output": {"keep": False, "confidence": 0.99, "reason": "zebrafish animal model"}},
    {"title": "Microbiota of homemade tepache fermented beverage",
     "output": {"keep": False, "confidence": 0.98, "reason": "food fermentation not human study"}},
    {"title": "Shotgun metagenomics of gut microbiome in type 2 diabetes",
     "output": {"keep": True, "confidence": 0.96, "reason": "human T2D metagenomics study"}},
]


class LLMVerifier:
    """Ollama-based verifier for borderline papers. Caches all results."""

    def __init__(self):
        self._cache = self._load_cache()
        self._available = False
        self._setup()

    def _setup(self):
        """Check Ollama is reachable and the model is available."""
        try:
            resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
            if resp.status_code == 200:
                models = [m["name"] for m in resp.json().get("models", [])]
                # Accept model name with or without tag suffix
                model_base = OLLAMA_MODEL.split(":")[0]
                available = any(m.split(":")[0] == model_base for m in models)
                if available:
                    self._available = True
                    logger.info(f"[llm_verifier] Ollama ready | model={OLLAMA_MODEL} | url={OLLAMA_BASE_URL}")
                else:
                    logger.warning(
                        f"[llm_verifier] Model '{OLLAMA_MODEL}' not found in Ollama. "
                        f"Available: {models}. Run: ollama pull {OLLAMA_MODEL}"
                    )
            else:
                logger.warning(f"[llm_verifier] Ollama returned HTTP {resp.status_code}")
        except requests.exceptions.ConnectionError:
            logger.warning(
                f"[llm_verifier] Cannot connect to Ollama at {OLLAMA_BASE_URL}. "
                "Borderline papers will go to review queue. "
                "Start Ollama with: ollama serve"
            )
        except Exception as e:
            logger.warning(f"[llm_verifier] Ollama setup check failed: {e}")

    @property
    def is_available(self) -> bool:
        return self._available

    # ── Main method ───────────────────────────────────────────────────────────

    def verify(self, title: str, abstract: Optional[str]) -> LLMVerdict:
        """Classify one paper. Returns cached result if available."""
        key = self._cache_key(title, abstract)

        # Cache hit — never call API again for same paper
        if key in self._cache:
            logger.debug(f"[llm_cache] hit {key[:8]}")
            c = self._cache[key]
            return LLMVerdict(keep=c["keep"], confidence=c["confidence"],
                              reason=c["reason"], cached=True)

        if not self._available:
            return LLMVerdict(keep=False, confidence=0.5,
                              reason="llm_unavailable_review_queue")

        try:
            verdict = self._call_ollama(title, abstract or "")

            self._cache[key] = {
                "keep": verdict.keep,
                "confidence": verdict.confidence,
                "reason": verdict.reason,
                "title": title[:80],
            }
            self._save_cache()
            return verdict

        except Exception as e:
            logger.error(f"[llm_verifier] API error: {e}")
            self._cache[key] = {
                "keep": False, "confidence": 0.5,
                "reason": "ollama_error", "title": title[:80],
            }
            self._save_cache()
            return LLMVerdict(keep=False, confidence=0.5, reason="ollama_error")

    # ── Ollama call ───────────────────────────────────────────────────────────

    def _build_prompt(self, title: str, abstract: str) -> str:
        examples = "\n".join([
            f"Title: {ex['title']}\nOutput: {json.dumps(ex['output'])}"
            for ex in FEW_SHOT_EXAMPLES
        ])
        return (
            f"{SYSTEM_PROMPT}\n\n"
            f"Examples:\n{examples}\n\n"
            f"Classify this paper:\n"
            f"Title: {title}\n"
            f"Abstract: {abstract[:500]}\n\n"
            f"Return JSON only."
        )

    def _call_ollama(self, title: str, abstract: str) -> LLMVerdict:
        prompt = self._build_prompt(title, abstract)
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.1},
        }
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=OLLAMA_TIMEOUT,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "{}")
        return self._parse(raw)

    # ── Response parser ───────────────────────────────────────────────────────

    def _parse(self, raw: str) -> LLMVerdict:
        """Parse JSON response, handle malformed output gracefully."""
        try:
            clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            data = json.loads(clean)
            return LLMVerdict(
                keep=bool(data.get("keep", False)),
                confidence=float(data.get("confidence", 0.5)),
                reason=str(data.get("reason", ""))[:100],
            )
        except Exception as e:
            logger.warning(f"[llm_verifier] JSON parse failed: {raw[:60]} | {e}")
            raw_l = raw.lower()
            if '"keep":true' in raw_l or '"keep": true' in raw_l:
                return LLMVerdict(keep=True, confidence=0.7, reason="fallback_keep")
            return LLMVerdict(keep=False, confidence=0.7, reason="fallback_reject")

    # ── Cache ─────────────────────────────────────────────────────────────────

    def _cache_key(self, title: str, abstract: Optional[str]) -> str:
        return hashlib.md5(f"{title}|{abstract or ''}".encode()).hexdigest()

    def _load_cache(self) -> dict:
        if CACHE_PATH.exists():
            try:
                with open(CACHE_PATH) as f:
                    c = json.load(f)
                logger.info(f"[llm_verifier] Cache: {len(c)} verdicts loaded")
                return c
            except Exception:
                return {}
        return {}

    def _save_cache(self):
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_PATH, "w") as f:
            json.dump(self._cache, f, indent=2)

    def cache_stats(self) -> dict:
        return {"total_cached": len(self._cache), "cache_path": str(CACHE_PATH)}
