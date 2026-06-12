"""
nlp/pipeline.py
----------------
Orchestrates all 5 NLP modules into one processing pass per paper.

HOW IT WORKS:
  For each PaperRecord from Layer 1:
    1. ArticleClassifier  → normalized article type + confidence
    2. JournalClassifier  → impact factor, quartile, field
    3. SectionParser      → abstract split into sections (+ full text if OA)
    4. NERExtractor       → taxa, diseases, methods, body sites, treatments
    5. DataAvailability   → accession numbers, repos, status

  All outputs merge into one EnrichedPaperRecord.
  Saves results to data/processed/enriched_YYYYMMDD_HHMMSS.json
  Passes the enriched list to Layer 3.

PERFORMANCE NOTES:
  - Rule-based modules (classifier, section parser, data_availability):
    ~1ms per paper — can process 1000 papers in ~1 second
  - NER rule-based:
    ~5ms per paper
  - NER with BioBERT model:
    ~500ms per paper on CPU — set use_ner_model=False for faster runs
    ~50ms per paper on GPU
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional
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
from nlp.fulltext.fulltext_orchestrator import (FullTextOrchestrator)
from nlp.study_design import (extract_design)
from nlp.evidence_extractor import (extract as extract_evidence)
from nlp.quality_scorer import (score as quality_score)

# Lazy import to avoid circular deps — EntityNormalizer is only needed at runtime
def _get_entity_normalizer():
    from graph.entity_normalizer import EntityNormalizer
    return EntityNormalizer()

class NLPPipeline:
    """
    Processes a list of PaperRecords through all 5 NLP modules.
    Produces a list of EnrichedPaperRecords ready for Layer 3.
    """

    def __init__(self, use_ner_model: bool = False, use_llm: bool = False):
        """
        use_ner_model: Whether to load BioBERT for NER (Tier 2).
                       False = rules only (fast, no GPU needed)
                       True  = rules + BioBERT (better recall, needs ~4GB RAM or GPU)
        use_llm: Whether to use Ollama LLM for Tier 3 entity extraction.
                 Requires Ollama running with a model loaded.
        """
        logger.info("Initializing NLP pipeline modules...")
        self.article_classifier  = ArticleClassifier()
        self.journal_classifier  = JournalClassifier()
        self.ner_extractor       = NERExtractor(use_model=use_ner_model, use_llm=use_llm)
        self.section_parser      = SectionParser()
        self.data_av_extractor   = DataAvailabilityExtractor()
        self.fulltext            = FullTextOrchestrator()
        # Entity normalizer for inline grounding at Layer 2 time
        # Loaded once and reused across all papers (SQLite cache amortizes API calls)
        try:
            self.entity_normalizer = _get_entity_normalizer()
            logger.info("[pipeline] EntityNormalizer loaded — inline grounding enabled")
        except Exception as e:
            self.entity_normalizer = None
            logger.warning(f"[pipeline] EntityNormalizer unavailable — grounding deferred to Layer 3: {e}")
        logger.success("NLP pipeline ready")

    def process_all(
        self,
        papers: List[PaperRecord],
        use_ner_model: bool = False,
    ) -> List[EnrichedPaperRecord]:
        """
        Processes all papers and returns enriched records.
        Saves to disk automatically.
        """
        logger.info(f"Starting NLP processing for {len(papers)} papers")
        enriched = []
        errors = 0

        for paper in tqdm(papers, desc="NLP processing"):
            try:
                result = self.process_one(paper)
                enriched.append(result)
            except Exception as e:
                logger.error(f"[pipeline] Failed on paper '{paper.title[:60]}': {e}")
                errors += 1
                continue

        logger.success(f"NLP complete: {len(enriched)} enriched, {errors} errors")
        output_path = self._save(enriched)
        logger.success(f"Saved → {output_path}")
        self._print_summary(enriched)
        return enriched

    def process_one(self, paper: PaperRecord) -> EnrichedPaperRecord:
        """
        Runs all 5 NLP modules on one paper.

        WHAT HAPPENS STEP BY STEP:
          1. Start with all fields from the PaperRecord
          2. Module 1 reads article_types + title + abstract → normalized type
          3. Module 2 reads journal name + ISSN → impact factor, quartile
          4. Module 3 (section parser) reads abstract → ParsedSection list
          5. Module 4 (NER) reads title + abstract → entity list + grouped dict
          6. Module 5 reads sections → DataAvailabilityInfo
          7. Everything merges into EnrichedPaperRecord
        """
        # ── Full text acquisition ──

        full = (self.fulltext.fetch(paper) or {})

        full_text = (full.get("full_text","")
        or
        " ".join([
            full.get("abstract",""),
            full.get("methods",""),
            full.get("results",""),
            full.get("discussion","")]))
        full_text = full_text.strip()
        

        # ── Module 1: Article type classification ─────────────────────────────
        article_type, confidence = self.article_classifier.classify(
            article_types_raw=paper.article_types,
            title=paper.title,
            abstract=paper.abstract,
        )

        # ── Module 2: Journal classification ─────────────────────────────────
        journal_info: JournalInfo = self.journal_classifier.classify(
            journal_name=paper.journal,
            issn=paper.issn,
        )

        # ── Module 3: Section parsing ─────────────────────────────────────────
        sections = self.section_parser.parse_abstract(paper.abstract)

        if full_text:

            full_sections = (

                self.section_parser.parse_full_text(full_text))

            sections.extend(full_sections)
        # If the paper is open access and we have full text, parse that too
        # (full text fetching will be added in a later enhancement)

        # ── Module 4: NER extraction ──────────────────────────────────────────
        entities = self.ner_extractor.extract(
            title=paper.title,
            abstract=paper.abstract,
            sections=sections if sections else None,
            full_text=full_text if full_text else None,
        )
        grouped = self.ner_extractor.group_entities(entities)

        # ── Module 4b: Inline entity grounding ───────────────────────────────
        # Ground each extracted entity to its canonical ontology ID here in Layer 2
        # so the enriched record is self-contained and Layer 3 doesn't re-normalize.
        # The grounding cache (SQLite) ensures APIs are only called once per entity.
        if self.entity_normalizer is not None:
            grounded_entities = []
            for ent in entities:
                try:
                    result = self.entity_normalizer.normalize(ent.text, ent.label)
                    grounded_entities.append(ent.model_copy(update={
                        "canonical_name":       result.get("canonical_name") or ent.text,
                        "ontology_id":          result.get("id"),
                        "ontology_name":        result.get("ontology"),
                        "grounded":             result.get("grounded", False),
                        "grounding_confidence": result.get("confidence", 0.0),
                        "grounding_source":     result.get("source", "none"),
                    }))
                except Exception:
                    grounded_entities.append(ent)
            entities = grounded_entities

        # ── Module 5: Data availability extraction ────────────────────────────
        data_availability: DataAvailabilityInfo = self.data_av_extractor.extract(sections=sections,abstract=paper.abstract,)

        
        source_text = (
            full_text
            if full_text and full_text.strip()
            else
            paper.abstract)
        study_design = (extract_design(source_text))
        evidence = (extract_evidence(source_text))


        # ── Merge everything into EnrichedPaperRecord ─────────────────────────
        # Pop 'full_text' from the base dump so the explicit kwarg below doesn't
        # collide with the field already present in PaperRecord.model_dump().
        paper_fields = paper.model_dump()
        paper_fields.pop("full_text", None)

        enriched = EnrichedPaperRecord(
            **paper_fields,                    # All Layer 1 fields carried forward
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
            # ── New 12 entity group fields ────────────────────────────────────
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
            nlp_version="1.0",
            full_text=full_text,
            fetch_source=full.get("fetch_source"),
            fetch_status=full.get("fetch_status"),
            study_design=study_design,
            evidence_score=evidence.get("sample_size",0),
            datasets=evidence.get("datasets",[]),
            quality_score=quality_score({
                    "journal_info":

                    journal_info,

                    "article_type":

                    article_type,

                    "data_availability":

                    data_availability,

                    "study_design":

                    study_design,

                    "sample_size":

                    evidence.get(
                        "sample_size",
                        0
                    )

                }

            ))

        return enriched

    def load_latest(self) -> List[EnrichedPaperRecord]:
        """Loads the most recent enriched output file from disk."""
        files = sorted(PROC_DIR.glob("enriched_*.json"), reverse=True)
        if not files:
            raise FileNotFoundError("No enriched data found. Run process_all() first.")
        path = files[0]
        logger.info(f"Loading: {path}")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return [EnrichedPaperRecord(**item) for item in data]

    def _save(self, enriched: List[EnrichedPaperRecord]) -> Path:
        """Saves enriched records to a timestamped JSON file."""
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        path = PROC_DIR / f"enriched_{ts}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump([r.model_dump() for r in enriched], f, indent=2,
                      ensure_ascii=False, default=str)
        return path

    def _print_summary(self, enriched: List[EnrichedPaperRecord]):
        """Prints a human-readable summary of Layer 2 results."""
        from collections import Counter

        types   = Counter(r.article_type_normalized for r in enriched)
        q_dist  = Counter(
            (r.journal_info.quartile if r.journal_info else "unknown")
            for r in enriched
        )
        da_dist = Counter(
            (r.data_availability.status if r.data_availability else "not_stated")
            for r in enriched
        )
        oa_count = sum(
            1
            for r in enriched
            if (r.journal_info and getattr(r.journal_info,"is_open_access",False)))

        logger.info("─" * 40)
        logger.info("NLP PIPELINE SUMMARY")
        logger.info(f"  Total enriched:         {len(enriched)}")
        logger.info(f"  Article types:          {dict(types.most_common())}")
        logger.info(f"  Journal quartiles:      {dict(q_dist.most_common())}")
        logger.info(f"  Data availability:      {dict(da_dist.most_common())}")
        logger.info(f"  Open access papers:     {oa_count}")
        logger.info("─" * 40)
