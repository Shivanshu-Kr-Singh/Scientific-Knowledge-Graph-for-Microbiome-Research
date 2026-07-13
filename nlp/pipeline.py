"""
nlp/pipeline.py
----------------
Orchestrates all 5 NLP modules into one processing pass per paper.

SCALE OPTIMIZATIONS (v2.0):
  1. Parallel processing   — ProcessPoolExecutor, configurable workers
  2. Incremental processing — skips already-processed papers by content_hash
  3. Chunked output        — writes enriched_batch_NNN.json (1000 per file)
                             + enriched_manifest.json tracking all batches
  4. load_all()            — merges all enriched batch files across runs

HOW IT WORKS:
  For each PaperRecord from Layer 1:
    1. ArticleClassifier  → normalized article type + confidence
    2. JournalClassifier  → impact factor, quartile, field
    3. SectionParser      → abstract split into sections
    4. NERExtractor       → taxa, diseases, methods, body sites, treatments
    5. DataAvailability   → accession numbers, repos, status

PERFORMANCE:
  Rule-based modules: ~1ms/paper | NER rules: ~5ms/paper
  NER BioBERT: ~500ms/paper CPU, ~50ms GPU (use_ner_model=False default)
  Parallel: up to NLP_WORKERS subprocesses (default = min(cpu_count, 8))
"""

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import List, Set
from loguru import logger
from tqdm import tqdm

from config import PROC_DIR
from models import PaperRecord
from nlp.enriched_record import EnrichedPaperRecord, DataAvailabilityInfo, JournalInfo
from nlp.article_classifier import ArticleClassifier
from nlp.journal_classifier import JournalClassifier
from nlp.ner import NERExtractor
from nlp.section_parser import SectionParser
from nlp.data_availability import DataAvailabilityExtractor
from nlp.fulltext.fulltext_orchestrator import FullTextOrchestrator
from nlp.study_design import extract_design
from nlp.evidence_extractor import extract as extract_evidence
from nlp.quality_scorer import score as quality_score

CHUNK_SIZE   = int(os.getenv("NLP_CHUNK_SIZE", "5000"))
MAX_WORKERS  = int(os.getenv("NLP_WORKERS", str(min(os.cpu_count() or 4, 8))))

# Detect whether a GPU is available for BioBERT inference.
# When GPU is available AND use_ner_model=True, we switch Phase 2 from
# ProcessPoolExecutor to ThreadPoolExecutor — one shared GPU process
# instead of 8 separate processes competing for GPU memory.
def _gpu_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

GPU_AVAILABLE = _gpu_available()
SKIP_FULLTEXT = os.getenv("NLP_SKIP_FULLTEXT", "false").lower() == "true"  # set true for speed runs
MANIFEST_PATH = PROC_DIR / "enriched_manifest.json"
HASH_INDEX_PATH = PROC_DIR / "enriched_hashes.txt"   # lightweight hash index


def _get_entity_normalizer():
    from graph.entity_normalizer import EntityNormalizer
    return EntityNormalizer()


# ── Phase 1 worker: full text fetch (I/O — runs in ThreadPoolExecutor) ────────

# Module-level shared orchestrator for Phase 1 — all threads reuse this
# single instance so they share the same in-memory fetch cache. Without this,
# each paper creates a new FullTextOrchestrator(), loads the 5-entry cache
# from disk, adds 1 entry, then gets garbage collected without saving —
# meaning no thread ever sees another thread's results, and nothing is
# persisted. With a shared instance, cache hits compound within the run.
_SHARED_ORCHESTRATOR = None
_ORCHESTRATOR_LOCK = None


def _get_shared_orchestrator():
    """Lazily creates the shared FullTextOrchestrator singleton."""
    global _SHARED_ORCHESTRATOR, _ORCHESTRATOR_LOCK
    import threading
    if _ORCHESTRATOR_LOCK is None:
        _ORCHESTRATOR_LOCK = threading.Lock()
    if _SHARED_ORCHESTRATOR is None:
        with _ORCHESTRATOR_LOCK:
            if _SHARED_ORCHESTRATOR is None:
                from nlp.fulltext.fulltext_orchestrator import FullTextOrchestrator
                _SHARED_ORCHESTRATOR = FullTextOrchestrator()
    return _SHARED_ORCHESTRATOR


def _fetch_fulltext_worker(paper_dict: dict) -> dict:
    """
    Fetches full text for one paper using all available strategies.
    Runs in a thread (not process) because it's pure network I/O —
    threads release the GIL during socket waits, allowing true concurrency.

    Returns the paper_dict with 'full_text', 'fetch_source', 'fetch_status'
    added so Phase 2 (CPU NLP) can use it without re-fetching.
    """
    from models import PaperRecord

    paper = PaperRecord(**paper_dict)
    try:
        full = _get_shared_orchestrator().fetch(paper) or {}
    except Exception:
        full = {}

    paper_dict["_full"]      = full
    paper_dict["_full_text"] = (
        full.get("full_text", "") or
        " ".join([
            full.get("abstract", ""),
            full.get("methods", ""),
            full.get("results", ""),
            full.get("discussion", ""),
        ])
    ).strip()

    try:
        full = FullTextOrchestrator().fetch(paper) or {}
    except Exception:
        full = {}

    full_text = (
        full.get("full_text", "") or
        " ".join([
            full.get("abstract", ""),
            full.get("methods", ""),
            full.get("results", ""),
            full.get("discussion", ""),
        ])
    ).strip()

    paper_dict["_full"]      = full
    paper_dict["_full_text"] = full_text
    return paper_dict


# ── Phase 2 worker: CPU NLP (runs in ProcessPoolExecutor) ────────────────────

# ── Process-local module cache for Phase 2 workers ───────────────────────────
# Each ProcessPoolExecutor worker handles many papers in sequence. Without
# caching, every paper re-instantiates NERExtractor (which reloads the 440MB
# BioBERT model from disk — ~1-2s overhead per paper, multiplied by 3,000+
# papers). This dict persists across papers within a single worker process.
_WORKER_MODULES: dict = {}


def _get_worker_modules(use_ner_model: bool, use_llm: bool) -> dict:
    """
    Returns cached NLP module instances for the current worker process.
    First call loads everything (including BioBERT if enabled); subsequent
    calls return the same instances instantly.
    """
    global _WORKER_MODULES
    if _WORKER_MODULES:
        return _WORKER_MODULES

    from nlp.article_classifier import ArticleClassifier
    from nlp.journal_classifier import JournalClassifier
    from nlp.ner import NERExtractor
    from nlp.section_parser import SectionParser
    from nlp.data_availability import DataAvailabilityExtractor

    _WORKER_MODULES = {
        "article_classifier": ArticleClassifier(),
        "journal_classifier": JournalClassifier(),
        "ner": NERExtractor(use_model=use_ner_model, use_llm=use_llm),
        "section_parser": SectionParser(),
        "data_availability": DataAvailabilityExtractor(),
    }
    return _WORKER_MODULES


def _process_one_worker(paper_dict: dict, use_ner_model: bool, use_llm: bool) -> dict:
    """
    Phase 2 worker: CPU-bound NLP (ArticleClassifier, NER, SectionParser etc).
    Runs in ProcessPoolExecutor. Full text pre-fetched by Phase 1 thread pool
    and stored in paper_dict['_full_text'] — no HTTP calls happen here.

    PERFORMANCE FIX: Heavy NLP objects (especially NERExtractor with BioBERT)
    are cached as process-local globals via _get_worker_modules(). Each worker
    process loads BioBERT exactly ONCE (on its first paper), then reuses it
    for all subsequent papers that process handles — avoiding the ~1-2s
    model-reload overhead that was previously incurred per paper.
    """
    from nlp.enriched_record import EnrichedPaperRecord
    from nlp.study_design import extract_design
    from nlp.evidence_extractor import extract as extract_evidence
    from nlp.quality_scorer import score as quality_score
    from models import PaperRecord
    from datetime import datetime

    # Get cached module instances (loaded once per process, reused across papers)
    modules = _get_worker_modules(use_ner_model, use_llm)

    paper = PaperRecord(**paper_dict)

    # Use pre-fetched full text from Phase 1 — no network calls needed
    full      = paper_dict.get("_full", {})
    full_text = paper_dict.get("_full_text", "") or (
        full.get("full_text", "") or
        " ".join([
            full.get("abstract", ""),
            full.get("methods", ""),
            full.get("results", ""),
            full.get("discussion", ""),
        ])
    ).strip()

    try:
        article_type, confidence = modules["article_classifier"].classify(
            article_types_raw=paper.article_types,
            title=paper.title,
            abstract=paper.abstract,
        )
        journal_info  = modules["journal_classifier"].classify(journal_name=paper.journal, issn=paper.issn)
        section_parser = modules["section_parser"]
        sections = section_parser.parse_abstract(paper.abstract)
        if full_text:
            sections.extend(section_parser.parse_full_text(full_text))

        ner = modules["ner"]
        entities = ner.extract(title=paper.title, abstract=paper.abstract,
                               sections=sections or None, full_text=full_text or None)
        grouped  = ner.group_entities(entities)

        data_availability = modules["data_availability"].extract(
            sections=sections, abstract=paper.abstract)

        src = full_text if full_text and full_text.strip() else paper.abstract
        study_design = extract_design(src)
        evidence     = extract_evidence(src)

        paper_fields = paper.model_dump()
        paper_fields.pop("full_text", None)

        # ── Store full text separately, keep only path in record ────────────
        from nlp.fulltext.fulltext_store import get_store
        ft_store = get_store()
        ft_path  = ft_store.save(paper.content_hash or "", full_text)

        enriched = EnrichedPaperRecord(
            **paper_fields,
            article_type_normalized=article_type,
            article_type_confidence=confidence,
            journal_info=journal_info,
            sections=sections,
            entities=entities,
            taxa=grouped.get("taxon", []),
            diseases=grouped.get("disease", []),
            methods=grouped.get("method", []),
            body_sites=grouped.get("body_site", []),
            treatments=grouped.get("treatment", []),
            metabolites=grouped.get("metabolite", []),
            genes=grouped.get("gene", []),
            proteins=grouped.get("protein", []),
            biomarkers=grouped.get("biomarker", []),
            pathways=grouped.get("pathway", []),
            populations=grouped.get("population", []),
            dietary_components=grouped.get("dietary_component", []),
            immune_cells=grouped.get("immune_cell", []),
            clinical_outcomes=grouped.get("clinical_outcome", []),
            environmental_factors=grouped.get("environmental_factor", []),
            sequencing_platforms=grouped.get("sequencing_platform", []),
            omics_features=grouped.get("omics_feature", []),
            other_entities=grouped.get("other_entities", {}),
            data_availability=data_availability,
            nlp_processed_at=datetime.utcnow().isoformat(),
            nlp_version="2.0",
            full_text=None,          # cleared — stored in fulltext_store
            fulltext_path=ft_path,   # path to data/fulltext/{hash}.txt
            fetch_source=full.get("fetch_source"),
            fetch_status=full.get("fetch_status"),
            study_design=study_design,
            evidence_score=evidence.get("sample_size", 0),
            datasets=evidence.get("datasets", []),
            quality_score=quality_score({
                "journal_info": journal_info,
                "article_type": article_type,
                "data_availability": data_availability,
                "study_design": study_design,
                "sample_size": evidence.get("sample_size", 0),
            }),
        )
        return enriched.model_dump()

    except Exception as e:
        return {"_error": str(e), "_title": paper.title[:80],
                "content_hash": paper.content_hash}


class NLPPipeline:
    """
    Processes PaperRecords through all 5 NLP modules.
    v2.0: parallel, incremental, chunked output, load_all().
    """

    def __init__(self, use_ner_model: bool = False, use_llm: bool = False):
        self.use_ner_model = use_ner_model
        self.use_llm       = use_llm

        logger.info("Initializing NLP pipeline modules...")
        self.article_classifier = ArticleClassifier()
        self.journal_classifier = JournalClassifier()
        self.ner_extractor      = NERExtractor(use_model=use_ner_model, use_llm=use_llm)
        self.section_parser     = SectionParser()
        self.data_av_extractor  = DataAvailabilityExtractor()
        self.fulltext           = FullTextOrchestrator()

        try:
            self.entity_normalizer = _get_entity_normalizer()
            logger.info("[pipeline] EntityNormalizer loaded — inline grounding enabled")
        except Exception as e:
            self.entity_normalizer = None
            logger.warning(f"[pipeline] EntityNormalizer unavailable: {e}")

        logger.success(
            f"NLP pipeline ready | workers={MAX_WORKERS} | "
            f"chunk_size={CHUNK_SIZE} | ner_model={use_ner_model}"
        )

    # ── Public interface ──────────────────────────────────────────────────────

    def process_all(self, papers: List[PaperRecord],
                    use_ner_model: bool = False) -> List[EnrichedPaperRecord]:
        """
        Processes all papers with parallel + incremental + chunked output.
        Steps:
          1. Load already-processed content_hashes from manifest
          2. Filter to only NEW papers
          3. Process in parallel, flush chunks to disk as they fill
          4. Return newly enriched records
        """
        logger.info(f"Starting NLP processing for {len(papers)} papers")

        done_hashes = self._load_processed_hashes()
        logger.info(f"[pipeline] Already processed: {len(done_hashes)} papers")

        new_papers = [p for p in papers if (p.content_hash or "") not in done_hashes]
        skipped    = len(papers) - len(new_papers)

        if skipped:
            logger.info(
                f"[pipeline] Skipping {skipped} already-processed | "
                f"Processing {len(new_papers)} new"
            )

        if not new_papers:
            logger.info("[pipeline] All papers already processed — loading from disk")
            return self.load_all()

        enriched_new = self._process_parallel(new_papers)
        self._print_summary(enriched_new)
        logger.success(
            f"NLP complete: {len(enriched_new)} newly enriched "
            f"({skipped} skipped)"
        )
        return enriched_new

    def _batch_resolve_pmcids(self, paper_dicts: List[dict]) -> None:
        """
        Pre-resolves DOI → PMCID for every paper lacking a PMCID, in batches
        of up to 200 per request, before Phase 1 full-text fetching starts.

        This is a throughput optimization only — it doesn't change which
        papers get a PMCID (same NCBI ID Converter, same coverage), it just
        avoids doing one HTTP round trip per paper. FullTextOrchestrator's
        per-paper resolve() call later reads from the same persistent cache
        this populates, so it becomes a free cache hit instead of a network
        call for every paper resolved here.

        Safe to call on any batch size — no-ops instantly if every paper
        already has a pmcid or lacks a doi.
        """
        dois_needing_lookup = [
            pd.get("doi")
            for pd in paper_dicts
            if pd.get("doi") and not pd.get("pmcid")
        ]
        if not dois_needing_lookup:
            return

        try:
            from nlp.fulltext.pmcid_resolver import PMCIDResolver
            resolver = PMCIDResolver()
            resolver.resolve_batch(dois_needing_lookup)
            resolver.flush_cache()
        except Exception as e:
            logger.warning(
                f"[pipeline] Batch PMCID resolution failed (non-fatal, "
                f"per-paper fallback still applies): {e}"
            )

    def _pmc_enrich(self, paper_dicts: List[dict]) -> None:
        """
        Fetches structured full text from PMC for all papers that have a PMCID
        but no full_text yet. Runs after PMCID resolution so newly-resolved
        PMCIDs are also enriched.

        Skips papers whose content_hash is already marked "success" in the
        full-text fetch cache (fetch_cache.json) — means their full text was
        already fetched in a previous run and is stored on disk. This prevents
        the PMC enricher from re-running the same 43 papers on every Layer 2
        restart.

        Modifies paper_dicts in-place — attaches full_text to each dict.
        """
        from collectors.pmc_enricher import PMCEnricher
        from models import PaperRecord
        from nlp.fulltext.fulltext_orchestrator import _load_cache as _load_fetch_cache

        # Load the full-text fetch cache to check what's already been fetched
        fetch_cache = _load_fetch_cache()

        candidates = [
            pd for pd in paper_dicts
            if pd.get("pmcid")
            and not pd.get("full_text")
            and fetch_cache.get(pd.get("content_hash", ""), {}).get("status") != "success"
        ]

        if not candidates:
            logger.info("[pmc_enricher] No papers with PMCID needing full text — skipping")
            return

        logger.info("")
        logger.info("=" * 60)
        logger.info(f"PMC FULL-TEXT ENRICHMENT — {len(candidates)} papers")
        logger.info("=" * 60)
        logger.info("")

        # Convert to PaperRecord for the enricher (it expects PaperRecord objects)
        paper_records = []
        for pd in candidates:
            try:
                paper_records.append(PaperRecord(**{
                    k: v for k, v in pd.items()
                    if not k.startswith("_")
                }))
            except Exception:
                paper_records.append(PaperRecord(
                    pmcid=pd.get("pmcid"),
                    title=pd.get("title", ""),
                    doi=pd.get("doi"),
                ))

        enricher = PMCEnricher()
        enriched_records = enricher.enrich(paper_records, max_enrichments=len(paper_records))

        # Map enriched full_text back into paper_dicts by pmcid
        enriched_map = {r.pmcid: r.full_text for r in enriched_records if r.full_text}

        count = 0
        for pd in paper_dicts:
            pmcid = pd.get("pmcid")
            if pmcid and pmcid in enriched_map and not pd.get("full_text"):
                pd["full_text"] = enriched_map[pmcid]
                count += 1
                # Mark this paper as "success" in the fetch cache so future
                # runs skip it rather than re-fetching from PMC every time.
                content_hash = pd.get("content_hash")
                if content_hash:
                    fetch_cache[content_hash] = {
                        "status": "success",
                        "fetch_source": "pmc_enricher",
                        "fetch_tier": 1,
                    }

        # Persist the updated fetch cache so next run sees these as done
        if count > 0:
            from nlp.fulltext.fulltext_orchestrator import _save_cache as _save_fetch_cache
            _save_fetch_cache(fetch_cache)

        logger.info(f"[pmc_enricher] Attached full text to {count} papers")

    def _process_parallel(self, papers: List[PaperRecord]) -> List[EnrichedPaperRecord]:
        """
        Two-phase parallel processing:

        Phase 1 — Full text fetch (ThreadPoolExecutor, I/O-bound):
          Threads release the GIL during network waits, so 32 threads can
          all be waiting on HTTP responses simultaneously. At 500K papers
          with ~20% having PMCID, this fetches ~100K full texts concurrently
          instead of sequentially.

        Phase 2 — CPU NLP (ProcessPoolExecutor, CPU-bound):
          Separate Python processes bypass the GIL for true CPU parallelism.
          NER regex, section parsing, article classification all run in
          parallel across MAX_WORKERS processes.

        Falls back to sequential if either executor fails.
        """
        from concurrent.futures import ThreadPoolExecutor

        enriched_all: List[EnrichedPaperRecord] = []
        chunk_buffer: List[dict] = []
        chunk_index  = self._next_chunk_index()
        errors       = 0
        paper_dicts  = [p.model_dump() for p in papers]

        # Number of threads for I/O phase — more threads than CPUs is fine for I/O
        io_workers = int(os.getenv("NLP_IO_WORKERS", str(min(len(paper_dicts), 64))))

        logger.info(
            f"[pipeline] {len(papers)} papers | "
            f"io_threads={io_workers} | cpu_workers={MAX_WORKERS} | "
            f"chunk_size={CHUNK_SIZE}"
        )

        # ── Phase 0: Batch-resolve DOI → PMCID before any fetching starts ──────
        # Collectors like Crossref, OpenAlex, and CORE never populate `pmcid`
        # even when the paper IS in PMC. FullTextOrchestrator resolves this
        # per-paper as a fallback, but doing it one DOI at a time doesn't
        # scale — at 10,000 papers that's ~1 hour serialized before Phase 1
        # even begins. Resolving the whole batch here first (up to 200 DOIs
        # per request) means every per-paper resolve() call below is just a
        # cache hit, not a new network round trip.
        self._batch_resolve_pmcids(paper_dicts)

        # ── Phase 0.5: PMC full-text enrichment ────────────────────────────────
        # Now that PMCIDs are resolved (both from collectors and DOI lookup),
        # fetch structured full text from PMC for all papers that have a PMCID
        # but no full_text yet. This was previously in Layer 1 but moved here
        # so newly-resolved PMCIDs also get full text.
        self._pmc_enrich(paper_dicts)

        # ── Phase 1: Fetch full text in parallel threads ───────────────────────
        logger.info("")
        logger.info("=" * 60)
        logger.info("PHASE 1 — FULL-TEXT ACQUISITION")
        logger.info("=" * 60)
        logger.info("")
        logger.info("[pipeline] Phase 1: fetching full text (ThreadPoolExecutor)...")
        enriched_dicts_with_ft: List[dict] = []

        try:
            with ThreadPoolExecutor(max_workers=io_workers) as thread_exec:
                ft_futures = {
                    thread_exec.submit(_fetch_fulltext_worker, pd): i
                    for i, pd in enumerate(paper_dicts)
                }
                with tqdm(total=len(ft_futures), desc="Fetching full text") as pbar:
                    for future in as_completed(ft_futures):
                        try:
                            enriched_dicts_with_ft.append(future.result())
                        except Exception as e:
                            # Keep the original dict without full text on failure
                            idx = ft_futures[future]
                            pd  = paper_dicts[idx]
                            pd["_full"]      = {}
                            pd["_full_text"] = ""
                            enriched_dicts_with_ft.append(pd)
                        pbar.update(1)

            ft_count = sum(1 for d in enriched_dicts_with_ft if d.get("_full_text"))
            logger.info(
                f"[pipeline] Phase 1 complete: {ft_count}/{len(papers)} "
                f"papers got full text"
            )
            # Flush the shared orchestrator's cache to disk so future runs
            # see these results as instant cache hits instead of re-fetching.
            try:
                _get_shared_orchestrator().flush_cache()
            except Exception:
                pass
        except Exception as e:
            logger.warning(
                f"[pipeline] Phase 1 (thread fetch) failed: {e} — "
                "continuing with abstract-only processing"
            )
            for pd in paper_dicts:
                pd["_full"]      = {}
                pd["_full_text"] = ""
            enriched_dicts_with_ft = paper_dicts

        # ── Phase 2: CPU NLP in parallel processes ─────────────────────────────
        logger.info("")
        logger.info("=" * 60)
        logger.info("PHASE 2 — NLP PROCESSING")
        logger.info("=" * 60)
        logger.info("")
        logger.info("[pipeline] Phase 2: NLP processing (ProcessPoolExecutor)...")

        # GPU MODE: when BioBERT is on GPU, use threads instead of processes.
        # Processes would split GPU memory across 8 workers = OOM crashes.
        # Threads share one process = one BioBERT on GPU + full VRAM.
        use_gpu_mode = self.use_ner_model and GPU_AVAILABLE

        if use_gpu_mode:
            logger.info(
                "[pipeline] GPU detected + BioBERT active — "
                "using ThreadPoolExecutor (shared GPU memory)"
            )
            return self._process_gpu_threaded(enriched_dicts_with_ft)

        try:
            with ProcessPoolExecutor(max_workers=MAX_WORKERS) as proc_exec:
                futures = {
                    proc_exec.submit(
                        _process_one_worker, pd,
                        self.use_ner_model, self.use_llm
                    ): pd.get("content_hash", "?")
                    for pd in enriched_dicts_with_ft
                }
                with tqdm(total=len(futures), desc="NLP processing") as pbar:
                    for future in as_completed(futures):
                        result = future.result()
                        pbar.update(1)
                        if "_error" in result:
                            logger.warning(
                                f"[pipeline] NLP failed: {result.get('_title','?')} "
                                f"— {result['_error']}"
                            )
                            errors += 1
                            continue
                        chunk_buffer.append(result)
                        if len(chunk_buffer) >= CHUNK_SIZE:
                            self._save_chunk(chunk_buffer, chunk_index)
                            enriched_all.extend(
                                EnrichedPaperRecord(**r) for r in chunk_buffer
                            )
                            chunk_index += 1
                            chunk_buffer = []

        except Exception as e:
            logger.warning(
                f"[pipeline] Phase 2 (process NLP) failed: {e} — "
                "falling back to sequential"
            )
            return self._process_sequential(papers)

        # Flush remaining partial chunk
        if chunk_buffer:
            self._save_chunk(chunk_buffer, chunk_index)
            enriched_all.extend(EnrichedPaperRecord(**r) for r in chunk_buffer)

        logger.info(f"[pipeline] Phase 2 complete | {errors} papers failed")
        return enriched_all

    def _process_gpu_threaded(self, paper_dicts: List[dict]) -> List[EnrichedPaperRecord]:
        """
        GPU-optimised processing path: uses ThreadPoolExecutor instead of
        ProcessPoolExecutor so all threads share one process and one GPU.

        BioBERT runs on the GPU in the main process via the shared
        self.ner_extractor instance. Other CPU-bound modules (section parser,
        article classifier, etc.) run in parallel threads.

        Architecture:
          - 1 shared NERExtractor instance (BioBERT on GPU)
          - NLP_WORKERS threads for I/O-bound modules (journal lookup, etc.)
          - BioBERT inference is GIL-released during CUDA ops → real concurrency
        """
        from concurrent.futures import ThreadPoolExecutor as ThreadExec

        enriched_all: List[EnrichedPaperRecord] = []
        chunk_buffer: List[dict] = []
        chunk_index  = self._next_chunk_index()
        errors       = 0

        # Number of threads — more than CPU cores since I/O dominates
        thread_workers = int(os.getenv("NLP_WORKERS", str(min(os.cpu_count() or 4, 16))))

        logger.info(
            f"[pipeline] GPU threaded: {len(paper_dicts)} papers | "
            f"threads={thread_workers}"
        )

        def process_one_threaded(pd: dict) -> dict:
            """Runs all NLP modules using the shared pipeline instance."""
            try:
                from models import PaperRecord
                paper     = PaperRecord(**pd)
                full      = pd.get("_full", {})
                full_text = pd.get("_full_text", "") or (
                    full.get("full_text", "") or
                    " ".join([
                        full.get("abstract", ""),
                        full.get("methods", ""),
                        full.get("results", ""),
                        full.get("discussion", ""),
                    ])
                ).strip()

                article_type, confidence = self.article_classifier.classify(
                    article_types_raw=paper.article_types,
                    title=paper.title,
                    abstract=paper.abstract,
                )
                journal_info   = self.journal_classifier.classify(
                    journal_name=paper.journal, issn=paper.issn
                )
                sections = self.section_parser.parse_abstract(paper.abstract)
                if full_text:
                    sections.extend(self.section_parser.parse_full_text(full_text))

                # Shared NER extractor — BioBERT runs on GPU in this process
                entities = self.ner_extractor.extract(
                    title=paper.title, abstract=paper.abstract,
                    sections=sections or None, full_text=full_text or None,
                )
                grouped  = self.ner_extractor.group_entities(entities)

                data_availability = self.data_av_extractor.extract(
                    sections=sections, abstract=paper.abstract,
                )
                src          = full_text if full_text and full_text.strip() else paper.abstract
                study_design = extract_design(src)
                evidence     = extract_evidence(src)

                from nlp.fulltext.fulltext_store import get_store
                ft_path      = get_store().save(paper.content_hash or "", full_text)

                paper_fields = paper.model_dump()
                paper_fields.pop("full_text", None)

                enriched = EnrichedPaperRecord(
                    **paper_fields,
                    article_type_normalized=article_type,
                    article_type_confidence=confidence,
                    journal_info=journal_info,
                    sections=sections,
                    entities=entities,
                    taxa=grouped.get("taxon", []),
                    diseases=grouped.get("disease", []),
                    methods=grouped.get("method", []),
                    body_sites=grouped.get("body_site", []),
                    treatments=grouped.get("treatment", []),
                    metabolites=grouped.get("metabolite", []),
                    genes=grouped.get("gene", []),
                    proteins=grouped.get("protein", []),
                    biomarkers=grouped.get("biomarker", []),
                    pathways=grouped.get("pathway", []),
                    populations=grouped.get("population", []),
                    dietary_components=grouped.get("dietary_component", []),
                    immune_cells=grouped.get("immune_cell", []),
                    clinical_outcomes=grouped.get("clinical_outcome", []),
                    environmental_factors=grouped.get("environmental_factor", []),
                    sequencing_platforms=grouped.get("sequencing_platform", []),
                    omics_features=grouped.get("omics_feature", []),
                    other_entities=grouped.get("other_entities", {}),
                    data_availability=data_availability,
                    nlp_processed_at=datetime.utcnow().isoformat(),
                    nlp_version="2.0",
                    full_text=None,
                    fulltext_path=ft_path,
                    fetch_source=full.get("fetch_source"),
                    fetch_status=full.get("fetch_status"),
                    study_design=study_design,
                    evidence_score=evidence.get("sample_size", 0),
                    datasets=evidence.get("datasets", []),
                    quality_score=quality_score({
                        "journal_info": journal_info,
                        "article_type": article_type,
                        "data_availability": data_availability,
                        "study_design": study_design,
                        "sample_size": evidence.get("sample_size", 0),
                        "sequencing_methods": evidence.get("sequencing_methods", []),
                    }),
                )
                return enriched.model_dump()
            except Exception as e:
                return {
                    "_error": str(e),
                    "_title": pd.get("title", "")[:80],
                    "content_hash": pd.get("content_hash"),
                }

        try:
            with ThreadExec(max_workers=thread_workers) as executor:
                futures = {
                    executor.submit(process_one_threaded, pd): pd.get("content_hash", "?")
                    for pd in paper_dicts
                }
                with tqdm(total=len(futures), desc="NLP processing (GPU)") as pbar:
                    for future in as_completed(futures):
                        result = future.result()
                        pbar.update(1)
                        if "_error" in result:
                            logger.warning(
                                f"[pipeline] Failed: {result.get('_title','?')} "
                                f"— {result['_error']}"
                            )
                            errors += 1
                            continue
                        chunk_buffer.append(result)
                        if len(chunk_buffer) >= CHUNK_SIZE:
                            self._save_chunk(chunk_buffer, chunk_index)
                            enriched_all.extend(
                                EnrichedPaperRecord(**r) for r in chunk_buffer
                            )
                            chunk_index  += 1
                            chunk_buffer  = []

        except Exception as e:
            logger.warning(
                f"[pipeline] GPU threaded processing failed ({e}) — "
                "falling back to sequential"
            )
            return self._process_sequential(
                [PaperRecord(**pd) for pd in paper_dicts]
            )

        if chunk_buffer:
            self._save_chunk(chunk_buffer, chunk_index)
            enriched_all.extend(EnrichedPaperRecord(**r) for r in chunk_buffer)

        logger.info(
            f"[pipeline] GPU threaded complete | "
            f"{len(enriched_all)} enriched | {errors} failed"
        )
        return enriched_all

    def _process_sequential(self, papers: List[PaperRecord]) -> List[EnrichedPaperRecord]:
        """Sequential fallback — used when multiprocessing is unavailable."""
        enriched_all: List[EnrichedPaperRecord] = []
        chunk_buffer: List[dict] = []
        chunk_index  = self._next_chunk_index()
        errors = 0

        for paper in tqdm(papers, desc="NLP processing (sequential)"):
            try:
                r = self.process_one(paper)
                chunk_buffer.append(r.model_dump())
                enriched_all.append(r)
                if len(chunk_buffer) >= CHUNK_SIZE:
                    self._save_chunk(chunk_buffer, chunk_index)
                    chunk_index += 1
                    chunk_buffer = []
            except Exception as e:
                logger.error(f"[pipeline] Failed: {paper.title[:60]} — {e}")
                errors += 1

        if chunk_buffer:
            self._save_chunk(chunk_buffer, chunk_index)
        logger.info(f"[pipeline] {errors} errors")
        return enriched_all

    def process_one(self, paper: PaperRecord) -> EnrichedPaperRecord:
        """Single-paper processing (sequential path / testing)."""
        # Smart orchestrator handles all strategies + caching — no shortcuts
        full = (self.fulltext.fetch(paper) or {})
        full_text = (
            full.get("full_text", "") or
            " ".join([full.get("abstract", ""), full.get("methods", ""),
                      full.get("results", ""), full.get("discussion", "")])
        ).strip()

        article_type, confidence = self.article_classifier.classify(
            article_types_raw=paper.article_types,
            title=paper.title, abstract=paper.abstract)
        journal_info = self.journal_classifier.classify(
            journal_name=paper.journal, issn=paper.issn)
        sections = self.section_parser.parse_abstract(paper.abstract)
        if full_text:
            sections.extend(self.section_parser.parse_full_text(full_text))

        entities = self.ner_extractor.extract(
            title=paper.title, abstract=paper.abstract,
            sections=sections or None, full_text=full_text or None)
        grouped  = self.ner_extractor.group_entities(entities)

        if self.entity_normalizer is not None:
            grounded = []
            for ent in entities:
                try:
                    res = self.entity_normalizer.normalize(ent.text, ent.label)
                    grounded.append(ent.model_copy(update={
                        "canonical_name":       res.get("canonical_name") or ent.text,
                        "ontology_id":          res.get("id"),
                        "ontology_name":        res.get("ontology"),
                        "grounded":             res.get("grounded", False),
                        "grounding_confidence": res.get("confidence", 0.0),
                        "grounding_source":     res.get("source", "none"),
                    }))
                except Exception:
                    grounded.append(ent)
            entities = grounded

        data_availability = self.data_av_extractor.extract(
            sections=sections, abstract=paper.abstract)

        src = full_text if full_text and full_text.strip() else paper.abstract
        study_design = extract_design(src)
        evidence     = extract_evidence(src)
        paper_fields = paper.model_dump()
        paper_fields.pop("full_text", None)

        return EnrichedPaperRecord(
            **paper_fields,
            article_type_normalized=article_type,
            article_type_confidence=confidence,
            journal_info=journal_info,
            sections=sections,
            entities=entities,
            taxa=grouped.get("taxon", []),
            diseases=grouped.get("disease", []),
            methods=grouped.get("method", []),
            body_sites=grouped.get("body_site", []),
            treatments=grouped.get("treatment", []),
            metabolites=grouped.get("metabolite", []),
            genes=grouped.get("gene", []),
            proteins=grouped.get("protein", []),
            biomarkers=grouped.get("biomarker", []),
            pathways=grouped.get("pathway", []),
            populations=grouped.get("population", []),
            dietary_components=grouped.get("dietary_component", []),
            immune_cells=grouped.get("immune_cell", []),
            clinical_outcomes=grouped.get("clinical_outcome", []),
            environmental_factors=grouped.get("environmental_factor", []),
            sequencing_platforms=grouped.get("sequencing_platform", []),
            omics_features=grouped.get("omics_feature", []),
            other_entities=grouped.get("other_entities", {}),
            data_availability=data_availability,
            nlp_processed_at=datetime.utcnow().isoformat(),
            nlp_version="2.0",
            full_text=None,          # cleared — stored in fulltext_store
            fulltext_path=(
                __import__('nlp.fulltext.fulltext_store', fromlist=['get_store'])
                .get_store().save(paper.content_hash or "", full_text)
            ),
            fetch_source=full.get("fetch_source"),
            fetch_status=full.get("fetch_status"),
            study_design=study_design,
            evidence_score=evidence.get("sample_size", 0),
            datasets=evidence.get("datasets", []),
            quality_score=quality_score({
                "journal_info": journal_info, "article_type": article_type,
                "data_availability": data_availability,
                "study_design": study_design,
                "sample_size": evidence.get("sample_size", 0),
            }),
        )

    # ── Loading ───────────────────────────────────────────────────────────────

    def load_latest(self) -> List[EnrichedPaperRecord]:
        """Loads the most recent enriched batch from manifest, or legacy file."""
        if MANIFEST_PATH.exists():
            manifest = self._load_manifest()
            if manifest.get("batches"):
                latest = sorted(manifest["batches"])[-1]
                path   = PROC_DIR / latest
                logger.info(f"Loading latest batch: {path}")
                with open(path, encoding="utf-8") as f:
                    return [EnrichedPaperRecord(**item) for item in json.load(f)]

        files = sorted(
            [f for f in PROC_DIR.glob("enriched_*.json")
             if "batch" not in f.name and "manifest" not in f.name],
            reverse=True)
        if not files:
            raise FileNotFoundError("No enriched data found. Run process_all() first.")
        with open(files[0], encoding="utf-8") as f:
            return [EnrichedPaperRecord(**item) for item in json.load(f)]

    def load_all(self) -> List[EnrichedPaperRecord]:
        """Merges ALL enriched batch files, deduplicating by content_hash."""
        manifest   = self._load_manifest()
        batch_files = manifest.get("batches", [])
        legacy = [f.name for f in sorted(PROC_DIR.glob("enriched_*.json"))
                  if "batch" not in f.name and "manifest" not in f.name]
        all_files = sorted(set(batch_files + legacy))

        if not all_files:
            raise FileNotFoundError("No enriched data found.")

        logger.info(f"[load_all] {len(all_files)} enriched file(s)")
        seen: Set[str] = set()
        all_enriched: List[EnrichedPaperRecord] = []
        total_raw = 0

        for fname in all_files:
            path = PROC_DIR / fname
            if not path.exists():
                continue
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                logger.warning(f"[load_all] Skipping {fname}: {e}")
                continue
            added = 0
            for item in data:
                key = item.get("content_hash") or item.get("doi") or item.get("pmid","")
                if key and key not in seen:
                    seen.add(key)
                    try:
                        all_enriched.append(EnrichedPaperRecord(**item))
                        added += 1
                    except Exception:
                        pass
            total_raw += len(data)
            logger.info(f"[load_all]   {fname}: {len(data)} raw, {added} new")

        logger.info(
            f"[load_all] Total: {total_raw} raw → {len(all_enriched)} unique"
        )
        return all_enriched

    # ── Chunk helpers ─────────────────────────────────────────────────────────

    def _save_chunk(self, records: List[dict], chunk_index: int) -> Path:
        filename = f"enriched_batch_{chunk_index:04d}.json"
        path     = PROC_DIR / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False, default=str)
        self._update_manifest(filename, len(records))
        self._append_hashes_to_index(records)   # keep flat index up to date
        logger.info(
            f"[pipeline] Chunk {chunk_index}: {len(records)} records → {filename}"
        )
        return path

    def _next_chunk_index(self) -> int:
        batches = self._load_manifest().get("batches", [])
        indices = []
        for b in batches:
            try:
                indices.append(int(b.replace("enriched_batch_","").replace(".json","")))
            except ValueError:
                pass
        return (max(indices) + 1) if indices else 0

    def _load_manifest(self) -> dict:
        if MANIFEST_PATH.exists():
            try:
                with open(MANIFEST_PATH, encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"batches": [], "total_records": 0}

    def _update_manifest(self, filename: str, count: int):
        manifest = self._load_manifest()
        if filename not in manifest["batches"]:
            manifest["batches"].append(filename)
        manifest["total_records"] = manifest.get("total_records", 0) + count
        manifest["last_updated"]  = datetime.utcnow().isoformat()
        with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    def _load_processed_hashes(self) -> Set[str]:
        """
        Returns content_hashes of already-enriched papers (incremental skip).

        Uses a lightweight flat text index (enriched_hashes.txt) — one hash
        per line. At 500K papers this is ~20MB vs reading 100 JSON files.
        Falls back to scanning batch files if index doesn't exist.
        """
        # Fast path: flat hash index
        if HASH_INDEX_PATH.exists():
            try:
                return set(HASH_INDEX_PATH.read_text(encoding="utf-8").splitlines())
            except Exception as e:
                logger.warning(f"[pipeline] Hash index read failed: {e}")

        # Slow path: scan batch files (first run or index missing)
        done: Set[str] = set()
        for fname in self._load_manifest().get("batches", []):
            path = PROC_DIR / fname
            if not path.exists():
                continue
            try:
                with open(path, encoding="utf-8") as f:
                    for item in json.load(f):
                        h = item.get("content_hash")
                        if h:
                            done.add(h)
            except Exception as e:
                logger.warning(f"[pipeline] Hash scan failed {fname}: {e}")

        # Write index so future runs are fast
        if done:
            try:
                HASH_INDEX_PATH.write_text("\n".join(done), encoding="utf-8")
                logger.info(f"[pipeline] Hash index written: {len(done)} hashes")
            except Exception:
                pass
        return done

    def _append_hashes_to_index(self, records: List[dict]):
        """Appends new content_hashes to the flat index after each chunk save."""
        new_hashes = [r["content_hash"] for r in records if r.get("content_hash")]
        if not new_hashes:
            return
        try:
            with open(HASH_INDEX_PATH, "a", encoding="utf-8") as f:
                f.write("\n".join(new_hashes) + "\n")
        except Exception as e:
            logger.warning(f"[pipeline] Hash index append failed: {e}")

    # ── Legacy save ───────────────────────────────────────────────────────────

    def _save(self, enriched: List[EnrichedPaperRecord]) -> Path:
        ts   = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        path = PROC_DIR / f"enriched_{ts}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump([r.model_dump() for r in enriched], f,
                      indent=2, ensure_ascii=False, default=str)
        return path

    def deduplicate_batches(self) -> int:
        """
        Deduplicates across all enriched_batch_*.json files in-place.

        WHY THIS IS NEEDED:
          load_all() deduplicates on load, but the batch files themselves
          can contain duplicates when the same paper appears in multiple
          Layer 1 collection runs (collected_*.json files). At 500K papers
          this bloats batch files and wastes disk space.

        WHAT IT DOES:
          1. Reads all batch files in order
          2. Keeps first occurrence of each content_hash
          3. Rewrites each batch file with only unique records
          4. Rebuilds the hash index

        Returns:
          Number of duplicate records removed across all batches.
        """
        manifest   = self._load_manifest()
        batch_files = manifest.get("batches", [])

        if not batch_files:
            logger.info("[pipeline] No batch files to deduplicate")
            return 0

        logger.info(f"[pipeline] Deduplicating {len(batch_files)} batch files...")
        seen:    Set[str] = set()
        removed: int      = 0

        for fname in batch_files:
            path = PROC_DIR / fname
            if not path.exists():
                continue
            try:
                with open(path, encoding="utf-8") as f:
                    records = json.load(f)
            except Exception as e:
                logger.warning(f"[pipeline] Cannot read {fname}: {e}")
                continue

            unique = []
            for rec in records:
                key = rec.get("content_hash") or rec.get("doi") or rec.get("pmid", "")
                if key and key not in seen:
                    seen.add(key)
                    unique.append(rec)
                else:
                    removed += 1

            if len(unique) != len(records):
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(unique, f, indent=2, ensure_ascii=False, default=str)
                logger.info(
                    f"[pipeline]   {fname}: {len(records)} → {len(unique)} "
                    f"({len(records)-len(unique)} removed)"
                )

        # Rebuild hash index from scratch after dedup
        if HASH_INDEX_PATH.exists():
            HASH_INDEX_PATH.unlink()
        self._load_processed_hashes()   # rebuilds index from batch files

        logger.success(
            f"[pipeline] Deduplication complete: {removed} duplicates removed "
            f"across {len(batch_files)} batch files"
        )
        return removed

    # ── Summary ───────────────────────────────────────────────────────────────

    def _print_summary(self, enriched: List[EnrichedPaperRecord]):
        from collections import Counter
        types   = Counter(r.article_type_normalized for r in enriched)
        q_dist  = Counter(
            (r.journal_info.quartile if r.journal_info else "unknown")
            for r in enriched)
        da_dist = Counter(
            (r.data_availability.status if r.data_availability else "not_stated")
            for r in enriched)
        oa = sum(1 for r in enriched
                 if r.journal_info and getattr(r.journal_info,"is_open_access",False))
        logger.info("─" * 40)
        logger.info("NLP PIPELINE SUMMARY")
        logger.info(f"  Total enriched:    {len(enriched)}")
        logger.info(f"  Article types:     {dict(types.most_common())}")
        logger.info(f"  Journal quartiles: {dict(q_dist.most_common())}")
        logger.info(f"  Data availability: {dict(da_dist.most_common())}")
        logger.info(f"  Open access:       {oa}")
        logger.info("─" * 40)
