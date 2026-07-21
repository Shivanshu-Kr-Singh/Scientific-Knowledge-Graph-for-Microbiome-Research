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
import math
import hashlib
import requests
from typing import List, Optional
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
from filelock import FileLock
from loguru import logger

from config import SEMANTIC_CACHE_THRESHOLD, EMBEDDING_STORE_DIR

# ── Config ──
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_VERIFIER_MODEL", os.getenv("OLLAMA_EXTRACTION_MODEL", "llama3"))
OLLAMA_TIMEOUT  = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "60"))

# Master switch for Stage 4 LLM verification in Layer 1.
# Set LLM_VERIFIER_ENABLED=false in .env to disable — borderline papers
# will go straight to the review queue without calling Ollama.
from config import LLM_VERIFIER_ENABLED

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
        if not LLM_VERIFIER_ENABLED:
            logger.info("[llm_verifier] DISABLED via LLM_VERIFIER_ENABLED=false in .env")
            return
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
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(self._cache, f, indent=2)

    def cache_stats(self) -> dict:
        return {"total_cached": len(self._cache), "cache_path": str(CACHE_PATH)}


# ── Semantic Cache ────────────────────────────────────────────────────────────


class SemanticCache:
    """
    Cosine-similarity cache for LLM verdicts.

    Reuses cached LLM verdicts for near-duplicate papers based on embedding
    similarity (threshold: 0.97). This is INDEPENDENT from the content-hash
    cache in data/processed/llm_cache.json.

    Storage:
      data/embeddings/llm_verdict_cache.npy       — (K, dim) float32 matrix
      data/embeddings/llm_verdict_cache_meta.json  — list of verdict metadata dicts

    Thread safety: Uses filelock for atomic read/write operations.
    """

    SIMILARITY_THRESHOLD = SEMANTIC_CACHE_THRESHOLD  # 0.97 from config

    def __init__(self, store_dir: Path | None = None):
        self._store_dir = Path(store_dir) if store_dir else EMBEDDING_STORE_DIR
        self._store_dir.mkdir(parents=True, exist_ok=True)

        self._npy_path = self._store_dir / "llm_verdict_cache.npy"
        self._meta_path = self._store_dir / "llm_verdict_cache_meta.json"
        self._lock_path = self._store_dir / "llm_verdict_cache.lock"

        self._vectors: np.ndarray = np.empty((0, 0), dtype=np.float32)
        self._metadata: List[dict] = []

        self._load()

    # ── Public API ────────────────────────────────────────────────────────

    def lookup(self, paper_embedding: np.ndarray) -> Optional[LLMVerdict]:
        """
        Check if a cached verdict exists for a near-duplicate paper.

        Computes cosine similarity between paper_embedding and all cached
        vectors. If max similarity > SIMILARITY_THRESHOLD (0.97), returns
        the cached LLMVerdict with cached=True.

        Parameters
        ----------
        paper_embedding : np.ndarray
            Shape (dim,) — the embedding of the candidate paper.

        Returns
        -------
        Optional[LLMVerdict]
            Cached verdict if a near-duplicate exists, otherwise None.
        """
        if self._vectors.size == 0 or len(self._metadata) == 0:
            return None

        query = paper_embedding.astype(np.float32).flatten()
        query_norm = np.linalg.norm(query)

        if query_norm == 0:
            return None

        # Cosine similarity against all cached vectors
        vector_norms = np.linalg.norm(self._vectors, axis=1)
        # Avoid division by zero for any zero-norm cached vectors
        safe_norms = np.where(vector_norms == 0, 1.0, vector_norms)

        similarities = np.dot(self._vectors, query) / (query_norm * safe_norms)

        max_idx = int(np.argmax(similarities))
        max_sim = float(similarities[max_idx])

        if max_sim > self.SIMILARITY_THRESHOLD:
            cached_meta = self._metadata[max_idx]
            verdict_data = cached_meta.get("verdict", {})
            logger.debug(
                f"[semantic_cache] Hit: similarity={max_sim:.4f} | "
                f"title='{cached_meta.get('title', '')[:50]}'"
            )
            return LLMVerdict(
                keep=bool(verdict_data.get("keep", False)),
                confidence=float(verdict_data.get("confidence", 0.5)),
                reason=str(verdict_data.get("reason", "semantic_cache_hit")),
                cached=True,
            )

        return None

    def store_verdict(
        self,
        paper_embedding: np.ndarray,
        verdict: LLMVerdict,
        paper,
    ) -> None:
        """
        Store a paper's embedding and its LLM verdict for future cache lookups.

        Parameters
        ----------
        paper_embedding : np.ndarray
            Shape (dim,) — the embedding of the verified paper.
        verdict : LLMVerdict
            The LLM verdict to cache.
        paper : PaperRecord (or any object with doi, pmid, title attributes)
            Paper metadata for traceability.
        """
        lock = FileLock(str(self._lock_path))

        with lock:
            # Reload from disk to pick up changes from other processes
            self._load()

            vec = paper_embedding.astype(np.float32).reshape(1, -1)

            if self._vectors.size == 0:
                self._vectors = vec
            else:
                self._vectors = np.vstack([self._vectors, vec])

            meta_entry = {
                "doi": getattr(paper, "doi", None),
                "pmid": getattr(paper, "pmid", None),
                "title": getattr(paper, "title", "")[:200],
                "verdict": {
                    "keep": verdict.keep,
                    "confidence": verdict.confidence,
                    "reason": verdict.reason,
                },
                "verified_at": datetime.now(timezone.utc).isoformat(),
            }
            self._metadata.append(meta_entry)

            # Persist to disk immediately
            self._save()

        logger.debug(
            f"[semantic_cache] Stored verdict: "
            f"doi={meta_entry['doi']} | keep={verdict.keep}"
        )

    @property
    def size(self) -> int:
        """Number of cached verdicts."""
        return len(self._metadata)

    def stats(self) -> dict:
        """Return cache statistics."""
        return {
            "cached_verdicts": len(self._metadata),
            "npy_path": str(self._npy_path),
            "meta_path": str(self._meta_path),
        }

    # ── Private Helpers ───────────────────────────────────────────────────

    def _load(self) -> None:
        """Load cached vectors and metadata from disk. Reinitializes on corruption."""
        # Load .npy vectors
        if self._npy_path.exists():
            try:
                vectors = np.load(str(self._npy_path), allow_pickle=False)
                if vectors.ndim != 2:
                    raise ValueError(f"Expected 2D array, got {vectors.ndim}D")
                self._vectors = vectors.astype(np.float32)
            except Exception as e:
                logger.error(
                    f"[semantic_cache] Corrupted .npy file: {e}. "
                    f"Reinitializing empty cache."
                )
                self._vectors = np.empty((0, 0), dtype=np.float32)
                self._metadata = []
                self._save()
                return
        else:
            self._vectors = np.empty((0, 0), dtype=np.float32)

        # Load metadata JSON
        if self._meta_path.exists():
            try:
                with open(self._meta_path, "r", encoding="utf-8") as f:
                    self._metadata = json.load(f)
                if not isinstance(self._metadata, list):
                    raise ValueError("Metadata must be a list")
            except Exception as e:
                logger.error(
                    f"[semantic_cache] Corrupted metadata file: {e}. "
                    f"Reinitializing empty cache."
                )
                self._vectors = np.empty((0, 0), dtype=np.float32)
                self._metadata = []
                self._save()
                return
        else:
            self._metadata = []

        # Consistency check: vectors and metadata must have same count
        if self._vectors.size > 0 and len(self._vectors) != len(self._metadata):
            logger.error(
                f"[semantic_cache] Vector/metadata count mismatch "
                f"({len(self._vectors)} vs {len(self._metadata)}). "
                f"Reinitializing empty cache."
            )
            self._vectors = np.empty((0, 0), dtype=np.float32)
            self._metadata = []
            self._save()

    def _save(self) -> None:
        """Persist vectors and metadata to disk."""
        if self._vectors.size > 0:
            np.save(str(self._npy_path), self._vectors)
        else:
            np.save(str(self._npy_path), np.empty((0, 0), dtype=np.float32))

        with open(self._meta_path, "w", encoding="utf-8") as f:
            json.dump(self._metadata, f, ensure_ascii=False, indent=2)


# ── Batched LLM Verifier ──────────────────────────────────────────────────────

@dataclass
class BatchVerdict:
    """One paper's verdict from a batch LLM call."""
    title: str
    keep: bool
    confidence: float
    reason: str


class BatchedVerifier:
    """
    Groups papers into batches of up to 16, sends structured JSON prompts to Ollama.
    Retry strategy: split failed batch in half, retry sub-batches.
    Falls back to single-paper for persistent failures → marks for human review.
    """
    MAX_BATCH_SIZE = 16

    def __init__(self):
        from config import BATCH_LLM_SIZE, BACKEND_CONFIG
        self._max_batch_size = min(BATCH_LLM_SIZE, self.MAX_BATCH_SIZE)
        self._timeout = BACKEND_CONFIG.ollama_timeout_seconds
        self._max_retries = BACKEND_CONFIG.ollama_max_retries
        self._base_url = OLLAMA_BASE_URL
        self._model = OLLAMA_MODEL

    def verify_batch(self, papers: List) -> List[BatchVerdict]:
        """
        Verify a list of papers in batches of ≤ MAX_BATCH_SIZE.
        Returns one BatchVerdict per input paper in the same order.
        """
        if not papers:
            return []

        # Split papers into batches
        batches = self._split_into_batches(papers)
        all_verdicts: List[BatchVerdict] = []
        total_success = 0
        total_retries = 0
        total_splits = 0

        for batch in batches:
            verdicts, retries, splits = self._process_batch(batch)
            all_verdicts.extend(verdicts)
            total_success += sum(1 for v in verdicts if v.reason != "HUMAN_REVIEW: parse failure")
            total_retries += retries
            total_splits += splits

        logger.info(
            f"[batched_verifier] Complete | "
            f"total_papers={len(papers)} | "
            f"batch_count={len(batches)} | "
            f"success_count={total_success} | "
            f"retry_count={total_retries} | "
            f"split_count={total_splits}"
        )

        return all_verdicts

    def _split_into_batches(self, papers: List) -> List[List]:
        """Split papers into batches of at most _max_batch_size."""
        batch_size = self._max_batch_size
        return [
            papers[i:i + batch_size]
            for i in range(0, len(papers), batch_size)
        ]

    def _process_batch(self, batch: List) -> tuple:
        """
        Process a single batch. Returns (verdicts, retry_count, split_count).
        On parse failure: split in half and retry sub-batches.
        On single-paper failure: mark for human review.
        """
        retry_count = 0
        split_count = 0

        logger.debug(f"[batched_verifier] Processing batch of {len(batch)} papers")

        # Try the full batch first
        for attempt in range(self._max_retries + 1):
            try:
                verdicts = self._call_ollama_batch(batch)
                if verdicts is not None:
                    return verdicts, retry_count, split_count
            except Exception as e:
                logger.warning(
                    f"[batched_verifier] Batch call failed (attempt {attempt + 1}): {e}"
                )
            retry_count += 1

        # Full batch failed — split in half and retry sub-batches
        if len(batch) == 1:
            # Single paper that still fails — mark for human review
            paper = batch[0]
            title = getattr(paper, "title", "Unknown")
            logger.warning(
                f"[batched_verifier] Single-paper retry failed, marking for human review: {title[:60]}"
            )
            return [
                BatchVerdict(
                    title=title,
                    keep=False,
                    confidence=0.0,
                    reason="HUMAN_REVIEW: parse failure",
                )
            ], retry_count, split_count

        # Split batch in half
        split_count += 1
        mid = len(batch) // 2
        left_batch = batch[:mid]
        right_batch = batch[mid:]

        logger.info(
            f"[batched_verifier] Splitting batch of {len(batch)} into "
            f"{len(left_batch)} + {len(right_batch)}"
        )

        left_verdicts, left_retries, left_splits = self._process_batch(left_batch)
        right_verdicts, right_retries, right_splits = self._process_batch(right_batch)

        return (
            left_verdicts + right_verdicts,
            retry_count + left_retries + right_retries,
            split_count + left_splits + right_splits,
        )

    def _call_ollama_batch(self, batch: List) -> Optional[List[BatchVerdict]]:
        """
        Send a batch of papers to Ollama as a structured JSON array prompt.
        Returns a list of BatchVerdict on success, or None if the response is unparseable.
        """
        prompt = self._build_batch_prompt(batch)
        payload = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0},
        }

        resp = requests.post(
            f"{self._base_url}/api/generate",
            json=payload,
            timeout=self._timeout,
        )
        resp.raise_for_status()

        raw = resp.json().get("response", "")
        return self._parse_batch_response(raw, batch)

    def _build_batch_prompt(self, batch: List) -> str:
        """Build a structured JSON array prompt for a batch of papers."""
        papers_json = []
        for paper in batch:
            papers_json.append({
                "title": getattr(paper, "title", ""),
                "abstract": (getattr(paper, "abstract", "") or "")[:800],
            })

        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"=== BATCH CLASSIFICATION ===\n"
            f"You are given {len(batch)} papers to classify. "
            f"Return a JSON object with a single key \"results\" containing an array of {len(batch)} verdicts, "
            f"one per paper in the SAME ORDER as the input.\n\n"
            f"Each verdict must have: {{\"keep\": true/false, \"confidence\": 0.0-1.0, \"reason\": \"one sentence\"}}\n\n"
            f"Input papers:\n{json.dumps(papers_json, indent=2)}\n\n"
            f"Return ONLY valid JSON in this format:\n"
            f"{{\"results\": [{{\"keep\": true/false, \"confidence\": 0.0-1.0, \"reason\": \"...\"}}]}}\n"
        )
        return prompt

    def _parse_batch_response(self, raw: str, batch: List) -> Optional[List[BatchVerdict]]:
        """
        Parse the JSON array response from Ollama.
        Returns list of BatchVerdict if successful, None if unparseable.
        """
        try:
            clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            data = json.loads(clean)

            # Handle response as {"results": [...]} or as a direct list [...]
            if isinstance(data, dict) and "results" in data:
                results = data["results"]
            elif isinstance(data, list):
                results = data
            else:
                logger.warning(f"[batched_verifier] Unexpected response structure: {raw[:100]}")
                return None

            if not isinstance(results, list):
                logger.warning(f"[batched_verifier] Results is not a list: {type(results)}")
                return None

            # If the number of results doesn't match batch size, it's unusable
            if len(results) != len(batch):
                logger.warning(
                    f"[batched_verifier] Result count mismatch: "
                    f"expected {len(batch)}, got {len(results)}"
                )
                return None

            verdicts = []
            for i, (result, paper) in enumerate(zip(results, batch)):
                title = getattr(paper, "title", "Unknown")
                try:
                    verdicts.append(BatchVerdict(
                        title=title,
                        keep=bool(result.get("keep", False)),
                        confidence=float(result.get("confidence", 0.5)),
                        reason=str(result.get("reason", ""))[:100],
                    ))
                except (TypeError, ValueError, AttributeError) as e:
                    logger.warning(
                        f"[batched_verifier] Failed to parse verdict {i} for '{title[:40]}': {e}"
                    )
                    return None

            return verdicts

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"[batched_verifier] JSON parse failed: {raw[:100]} | {e}")
            return None
