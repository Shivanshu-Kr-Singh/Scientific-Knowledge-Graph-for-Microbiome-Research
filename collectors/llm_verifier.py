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

SYSTEM_PROMPT = """/no_think
You are a biomedical literature relevance classifier for a scientific knowledge graph project.

PROJECT: Human Microbiome Research Literature Mining (2024-2026)
GOAL: Build a knowledge graph of human microbiome-disease-treatment associations.
Only include papers that contribute to this goal.

=== INCLUDE (keep: true) ===

REQUIRED — paper must study ALL of:
1. HUMAN SUBJECTS: patients, cohorts, volunteers, clinical trials, RCTs, observational studies,
   longitudinal studies, case-control, cross-sectional, meta-analyses, systematic reviews
   → Body sites: gut, oral, skin, vaginal, lung, nasal, blood, placenta, breast milk, urinary
   → Populations: adults, children, neonates, elderly, pregnant women, disease cohorts

2. MICROBIOME/MICROBIOTA: gut flora, microbial community, dysbiosis, microbial diversity,
   taxonomic profiling, microbial composition, microbial abundance, microbial function

3. AT LEAST ONE OF:
   - Sequencing: 16S rRNA, shotgun metagenomics, WGS, metatranscriptomics, amplicon sequencing
   - Bioinformatics: QIIME, MetaPhlAn, HUMAnN, DADA2, LEfSe, DESeq2
   - Outcomes: association with disease, intervention effect, clinical outcome, biomarker

ALWAYS INCLUDE:
- Reviews and meta-analyses of human microbiome studies
- Studies using FMT, probiotics, prebiotics, dietary interventions on microbiome
- Mechanistic studies (host-microbe interactions, metabolites, immune response)

=== EXCLUDE (keep: false) ===

REJECT if primary focus is ANY of:
- ANIMAL MODELS: mouse, rat, zebrafish, pig, dog, cattle, primate — even humanized/germ-free
  Exception: only reject if >50% of study is animal; if minor animal validation of human data, keep
- ENVIRONMENTAL: soil, marine, wastewater, sediment, rhizosphere, compost, aquatic ecosystems
- FOOD/FERMENTATION: kombucha, kefir, cheese, wine, beer, food microbiology, fermented products
- PLANT/AGRICULTURE: plant microbiome, crop microbiome, phytobiome, soil-plant interactions
- PURE PATHOGEN: single-pathogen infection with NO microbiome community analysis
  (e.g., H. pylori eradication without microbiome profiling is REJECT)
- IN VITRO ONLY: cell culture or ex vivo with no human or microbiome data

=== CONFIDENCE CALIBRATION ===
0.95-1.00: Explicit human microbiome + sequencing + disease/outcome
0.85-0.94: Strong human microbiome evidence, clear clinical relevance
0.70-0.84: Human microbiome likely but less explicit
0.50-0.69: Borderline — human and microbiome present but marginal relevance
0.00-0.49: REJECT

=== OUTPUT FORMAT ===
Return ONLY valid JSON, no markdown, no explanation:
{"keep": true/false, "confidence": 0.0-1.0, "reason": "one concise sentence",
 "human": true/false, "microbiome": true/false, "metagenomics": true/false,
 "animal": true/false, "environmental": true/false,
 "article_type": "RCT/cohort/case-control/cross-sectional/review/meta-analysis/preprint/other"}
"""

FEW_SHOT_EXAMPLES = [
    {
        "title": "Gut microbiome dysbiosis in IBD: 16S rRNA profiling of 200 patients",
        "abstract": "We performed 16S rRNA sequencing on fecal samples from 100 IBD patients and 100 healthy controls.",
        "output": {"keep": True, "confidence": 0.98, "reason": "human IBD cohort with 16S sequencing and disease association", "article_type": "case-control"}
    },
    {
        "title": "FMT improves outcomes in recurrent C. difficile infection: RCT",
        "abstract": "Double-blind RCT of fecal microbiota transplantation versus vancomycin in 120 patients.",
        "output": {"keep": True, "confidence": 0.97, "reason": "human RCT of FMT intervention with microbiome outcome", "article_type": "RCT"}
    },
    {
        "title": "Shotgun metagenomics reveals gut microbiome signatures in type 2 diabetes",
        "abstract": "Whole-metagenome sequencing of 300 T2D patients identifies Faecalibacterium prausnitzii depletion.",
        "output": {"keep": True, "confidence": 0.99, "reason": "human T2D metagenomics with specific taxon associations", "article_type": "cohort"}
    },
    {
        "title": "Butyrate produced by gut bacteria suppresses NF-kB signaling: mechanistic review",
        "abstract": "This review synthesizes evidence on microbial butyrate production and host immune modulation.",
        "output": {"keep": True, "confidence": 0.92, "reason": "mechanistic review of human gut microbiome-immune interactions", "article_type": "review"}
    },
    {
        "title": "Zebrafish gut microbiome response to antibiotics",
        "abstract": "We treated zebrafish with ampicillin and monitored gut microbiota changes by 16S sequencing.",
        "output": {"keep": False, "confidence": 0.99, "reason": "zebrafish animal model — no human data", "article_type": "other"}
    },
    {
        "title": "Helicobacter pylori eradication with triple therapy: clinical outcomes",
        "abstract": "We evaluated clarithromycin-based triple therapy for H. pylori eradication in 150 patients.",
        "output": {"keep": False, "confidence": 0.88, "reason": "single-pathogen infection treatment with no microbiome community profiling", "article_type": "RCT"}
    },
    {
        "title": "Soil microbiome diversity in agricultural fields",
        "abstract": "We characterized bacterial diversity in soil samples from 5 farming regions.",
        "output": {"keep": False, "confidence": 0.99, "reason": "environmental soil microbiome — not human study", "article_type": "other"}
    },
    {
        "title": "Vaginal microbiome in pregnancy and preterm birth risk: prospective cohort",
        "abstract": "16S rRNA sequencing of vaginal samples from 500 pregnant women assessed Lactobacillus dominance.",
        "output": {"keep": True, "confidence": 0.97, "reason": "human vaginal microbiome in pregnancy with clinical outcome", "article_type": "cohort"}
    },
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
            f"Title: {ex['title']}\nAbstract: {ex.get('abstract', '')[:150]}\nOutput: {json.dumps(ex['output'])}"
            for ex in FEW_SHOT_EXAMPLES
        ])
        return (
            f"{SYSTEM_PROMPT}\n\n"
            f"=== FEW-SHOT EXAMPLES ===\n{examples}\n\n"
            f"=== PAPER TO CLASSIFY ===\n"
            f"Title: {title}\n"
            f"Abstract: {abstract[:800]}\n\n"
            f"Return JSON only."
        )

    def _call_ollama(self, title: str, abstract: str) -> LLMVerdict:
        prompt = self._build_prompt(title, abstract)
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0},
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
