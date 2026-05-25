"""
Stage 4 — LLM verifier for borderline papers only.

SUPPORTED PROVIDERS:
gemini    → Google Gemini 2.0 Flash

CACHING:
Every LLM response is cached by content hash in data/processed/llm_cache.json.
Re-runs never call the API twice for the same paper — critical for:
-Cost control(free tier has daily limits)
-Reproducibility(same verdict every time — important for research)
"""
from tenacity import retry
from tenacity import stop_after_attempt
from tenacity import wait_exponential
import os
import json
import hashlib
from typing import Optional
from pathlib import Path
from dataclasses import dataclass
from loguru import logger

#── Config ──
LLM_PROVIDER  = os.getenv("LLM_PROVIDER", "gemini")
LLM_MODEL     = os.getenv("LLM_MODEL", "gemini-2.5-flash")
GEMINI_KEY    = os.getenv("GEMINI_API_KEY", "")

CACHE_PATH = Path(__file__).parent.parent / "data" / "processed" / "llm_cache.json"

@dataclass
class LLMVerdict:
    keep:        bool
    confidence:  float
    reason:      str
    cached:      bool = False
    cost_tokens: int  = 0


# ── Prompt (same for all providers) ───
SYSTEM_PROMPT = """
You are a biomedical literature relevance classifier for an academic research system:

PROJECT:
Literature Mining and Metagenomic Data Preprocessing for HUMAN MICROBIOME STUDIES (2024–2026)

TASK:
Determine whether a paper should be INCLUDED for downstream NLP, data extraction, and knowledge graph construction.

OUTPUT CLASS:

KEEP
→ Strongly relevant to human microbiome research.

REVIEW
→ Borderline / uncertain / partially relevant.

REJECT
→ Not relevant.

=========================================================
INCLUSION CRITERIA
=========================================================

KEEP if MOST evidence supports ALL major requirements:

[1] HUMAN RELEVANCE
Paper studies humans:
- patients
- cohorts
- volunteers
- clinical studies
- observational studies
- RCTs
- longitudinal studies
- case-control studies
- population studies
- maternal-infant studies

Human body sites include:
gut, intestinal, gastrointestinal, fecal, stool,
oral, salivary, skin, vaginal, lung, nasal,
blood, breast milk, placental, urogenital.

Reviews / meta-analyses are acceptable if primarily HUMAN microbiome research.

---------------------------------------------------------

[2] MICROBIOME RELEVANCE

Paper studies:

microbiome
microbiota
microbial community
gut flora
microbial ecology
taxonomic profiling
community composition

OR sequencing/profiling methods:

16S
shotgun metagenomics
metagenomics
metatranscriptomics
amplicon sequencing
microbial profiling
alpha diversity
beta diversity
taxonomic abundance
functional profiling

---------------------------------------------------------

[3] DATA / METAGENOMICS SIGNALS (HIGH PRIORITY)

Strong positive evidence:

sequencing
16S
shotgun
metagenome
metatranscriptome
SRA
PRJNA
ENA
MGnify
data availability
supplementary dataset
accession number
bioinformatics pipeline
profiling workflow

Presence of these increases confidence.

=========================================================
REJECT CONDITIONS
=========================================================

REJECT if PRIMARY focus is:

ANIMAL STUDY:
Any non-human animal model:
mouse, murine, rat, zebrafish,
fish, dog, cat, cattle, pig,
rabbit, sheep, goat, horse,
frog, insect, shrimp, camel,
primate, tortoise, etc.

IMPORTANT:
Animal study using HUMAN microbiota transplant
→ STILL REJECT.

---------------------------------------------------------

ENVIRONMENTAL MICROBIOME:

soil
rhizosphere
sediment
wastewater
marine
river
lake
wetland
mangrove
compost
bioreactor
aquatic ecosystem

---------------------------------------------------------

FOOD / FERMENTATION:

kombucha
tepache
kefir
cheese
wine
beer
fermentation
food microbiology

---------------------------------------------------------

PLANT / AGRICULTURAL MICROBIOME

plant microbiome
crop microbiome
agricultural microbiology

---------------------------------------------------------

PURE PATHOGEN / INFECTIOUS DISEASE PAPERS

Reject if ONLY infection/pathogen focus and NO microbiome profiling:

Examples:

Helicobacter pylori infection
antibiotic resistance only
protein screening only
single-pathogen studies
clinical infection reports

=========================================================
ARTICLE TYPE POLICY
=========================================================

KEEP:
review
systematic review
meta-analysis
RCT
cohort
case-control
cross-sectional
preprint with human data

REVIEW:
protocol papers
editorials
letters
perspectives
methods papers with unclear application

=========================================================
CONFIDENCE CALIBRATION
=========================================================

0.90–1.00:
Explicit human microbiome + sequencing/metagenomics

0.75–0.89:
Strong human microbiome evidence

0.50–0.74:
Borderline → REVIEW

0.00–0.49:
Reject

=========================================================
OUTPUT FORMAT
=========================================================

Return JSON ONLY.

{
    "decision":
        "keep" |
        "review" |
        "reject",

    "confidence":0.0-1.0,

    "reason":"short explanation",

    "human":true/false,

    "microbiome":true/false,

    "metagenomics":true/false,

    "animal":true/false,

    "environmental":true/false,

    "article_type":"review/cohort/RCT/preprint/etc"
}
"""

FEW_SHOT_EXAMPLES=[
{"title": "Gut microbiome in IBD patients: 16S rRNA cohort study",
    "output": {"keep": True,  "confidence": 0.97, "reason": "human IBD cohort with 16S sequencing"}},
{"title": "Zebrafish gut microbiome response to antibiotic treatment",
    "output": {"keep": False, "confidence": 0.99, "reason": "zebrafish animal model"}},
{"title": "Microbiota of homemade tepache fermented beverage",
    "output": {"keep": False, "confidence": 0.98, "reason": "food fermentation not human study"}},
{"title": "Shotgun metagenomics of gut microbiome in type 2 diabetes",
    "output": {"keep": True,  "confidence": 0.96, "reason": "human T2D metagenomics study"}},]


class LLMVerifier:
    """LLM-based verifier for borderline papers. Caches all results."""

    def __init__(self):
        self._cache  = self._load_cache()
        self._client = None
        self._available = False
        self._setup()

    def _setup(self):
        if LLM_PROVIDER == "gemini" and GEMINI_KEY:
            try:
                from google import genai
                client = genai.Client(api_key=GEMINI_KEY)
                self._client = client
                self._available = True
                logger.info(f"[llm_verifier] Gemini ready | model={LLM_MODEL} | FREE tier")
            except ImportError:
                logger.warning("[llm_verifier] Run: pip install google-generativeai")
            except Exception as e:
                logger.error(f"[llm_verifier] Gemini setup failed: {e}")

        else:
            logger.warning(
                f"[llm_verifier] No API key found for provider '{LLM_PROVIDER}'.\n"
                "  Set GEMINI_API_KEY in .env for free Gemini access.\n"
                "  Borderline papers will go to review queue."
            )

    @property
    def is_available(self) -> bool:
        return self._available


    # ── Main method ───
    def verify(self, title: str, abstract: Optional[str]) -> LLMVerdict:
        """Classify one paper. Returns cached result if available."""
        key = self._cache_key(title, abstract)

        # Cache hit — never call API again for same paper
        if key in self._cache:

            logger.debug(f"[llm_cache] hit {key[:8]}")
            c = self._cache[key]
            return LLMVerdict(keep=c["keep"],confidence=c["confidence"],reason=c["reason"],cached=True)

        if not self._available:
            return LLMVerdict(keep=False, confidence=0.5,
                              reason="llm_unavailable_review_queue")

        # Call API
        try:
            verdict = self._call_gemini(title, abstract or "")

            # Cache and return
            self._cache[key] = {
                "keep": verdict.keep, "confidence": verdict.confidence,
                "reason": verdict.reason, "title": title[:80]
            }
            self._save_cache()
            return verdict

        except Exception as e:

            logger.error(f"[llm_verifier] API error: {e}")

            self._cache[key] = {"keep": False,"confidence": 0.5,"reason": "quota_error","title": title[:80]}
            self._save_cache()
            return LLMVerdict(keep=False,confidence=0.5,reason="quota_error")


    # ── Provider implementations ───
    def _build_user_message(self, title: str, abstract: str) -> str:
        """Shared prompt construction for all providers."""
        examples = "\n".join([
            f"Title: {ex['title']}\nOutput: {json.dumps(ex['output'])}"
            for ex in FEW_SHOT_EXAMPLES])
        return(
            f"Examples:\n{examples}\n\n"
            f"Classify this paper:\n"
            f"Title: {title}\n"
            f"Abstract: {abstract[:500]}\n\n"
            f"Return JSON only.")


    @retry(stop=stop_after_attempt(5),wait=wait_exponential(multiplier=1,min=2,max=30))

    def _call_gemini(self, title: str, abstract: str) -> LLMVerdict:
        msg = self._build_user_message(title, abstract)
        response = self._client.models.generate_content(
            model=LLM_MODEL,
            contents=SYSTEM_PROMPT + "\n\n" + msg,
            config={"response_mime_type": "application/json",
                    "temperature": 0.1}
        )
        return self._parse(response.text.strip())


    # ── Response parser ───
    def _parse(self, raw: str, tokens: int = 0) -> LLMVerdict:
        """Parses JSON response, handles malformed output gracefully."""
        try:
            clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            data  = json.loads(clean)
            return LLMVerdict(
                keep=bool(data.get("keep", False)),
                confidence=float(data.get("confidence", 0.5)),
                reason=str(data.get("reason", ""))[:100],
                cost_tokens=tokens,
            )
        except Exception as e:
            logger.warning(f"[llm_verifier] JSON parse failed: {raw[:60]} | {e}")
            raw_l = raw.lower()
            if '"keep":true' in raw_l or '"keep": true' in raw_l:
                return LLMVerdict(keep=True,  confidence=0.7, reason="fallback_keep")
            return LLMVerdict(keep=False, confidence=0.7, reason="fallback_reject")


    # ── Cache ───
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
