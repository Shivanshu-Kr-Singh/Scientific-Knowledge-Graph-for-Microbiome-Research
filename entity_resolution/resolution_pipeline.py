"""
ResolutionPipeline — orchestrates the seven-strategy resolution sequence.

Wires all components together and exposes:
  - resolve(): full resolution with audit trail
  - normalize(): drop-in Spec 1 interface
  - batch_resolve(): batch resolution

Strategy sequence:
  1. ManualOverrideManager lookup
  2. Exact match against CanonicalRegistry
  3. Normalized match (case-fold, strip punctuation, collapse whitespace)
  4. AbbreviationExpander lookup — if match, re-enter from step 2 with
     expanded form (at most once, try each expansion in lexicographic order)
  5. SynonymIndex lookup
  6. FuzzyMatcher (edit distance ≤ 2, skip if < 4 code points)
  7. OntologyTraverser hierarchy search

Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 2.1, 2.2
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import List, Optional, Tuple

from entity_resolution.models import (
    CandidateScore,
    NormalizationResult,
    ResolutionRecord,
    ResolutionResult,
    ShadowModeDiscrepancy,
    UnresolvedEntity,
)
from entity_resolution.utils import normalize_surface_form

logger = logging.getLogger(__name__)

# Minimum grounding confidence threshold to accept a match (Requirement 1.2)
_CONFIDENCE_THRESHOLD = 0.5


class ResolutionPipeline:
    """
    Orchestrates the 7-strategy resolution sequence.

    All components are injected via the constructor so that tests can wire
    in-memory instances without touching the filesystem.

    Preconditions for resolve():
    - surface_form is a non-empty string (after whitespace trimming)
    - entity_type is one of: "taxon", "disease", "method"
    - paper_id is a non-empty string identifying the source paper

    Postconditions for resolve():
    - Returns ResolutionResult with all fields populated
    - winning_strategy is "none" if all strategies fail
    - canonical_id is None and grounded=False if unresolved
    - Abbreviation re-entry occurs at most once per call
    - A ResolutionRecord is written to the AuditStore (non-blocking)
    - ResolutionMetrics are updated for the current run
    - Result is stored in ResolutionCache with current registry_version

    Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 2.1, 2.2
    """

    def __init__(
        self,
        registry=None,
        override_manager=None,
        synonym_index=None,
        abbreviation_expander=None,
        fuzzy_matcher=None,
        ontology_traverser=None,
        ranking_function=None,
        audit_store=None,
        resolution_cache=None,
        metrics=None,
        shadow_mode: bool = False,
        spec1_normalizer=None,
    ) -> None:
        """
        Initialise the pipeline with injected components.

        All components are optional; missing components cause the corresponding
        strategy to be skipped gracefully.

        Args:
            registry:              CanonicalRegistry instance.
            override_manager:      ManualOverrideManager instance.
            synonym_index:         SynonymIndex instance.
            abbreviation_expander: AbbreviationExpander instance.
            fuzzy_matcher:         FuzzyMatcher instance.
            ontology_traverser:    OntologyTraverser instance.
            ranking_function:      RankingFunction instance.
            audit_store:           ResolutionAuditStore instance.
            resolution_cache:      ResolutionCache instance.
            metrics:               ResolutionMetrics instance.
            shadow_mode:           If True, run in shadow mode alongside Spec 1.
            spec1_normalizer:      Optional Spec 1 normalizer for shadow mode comparison.
                                   Must expose normalize(surface_form, entity_type) ->
                                   NormalizationResult. Required for full shadow mode
                                   discrepancy logging (wired in task 16.1).
        """
        self._registry = registry
        self._override_manager = override_manager
        self._synonym_index = synonym_index
        self._abbreviation_expander = abbreviation_expander
        self._fuzzy_matcher = fuzzy_matcher
        self._ontology_traverser = ontology_traverser
        self._shadow_mode = shadow_mode
        self._spec1_normalizer = spec1_normalizer

        # Use provided ranking function or create a default one
        if ranking_function is not None:
            self._ranking_function = ranking_function
        else:
            from entity_resolution.ranking_function import RankingFunction
            self._ranking_function = RankingFunction()

        self._audit_store = audit_store
        self._cache = resolution_cache
        self._metrics = metrics

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve(
        self,
        surface_form: str,
        entity_type: str,
        paper_id: str = "unknown",
    ) -> ResolutionResult:
        """
        Full resolution with audit trail.

        Executes the seven-strategy sequence in order, collecting all
        candidates into a conflict_set, then calls RankingFunction.rank()
        to select the winner.

        Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 2.1, 2.2
        """
        now = datetime.now(timezone.utc)

        # --- Cache check --------------------------------------------------
        if self._cache is not None and self._registry is not None:
            registry_version = self._registry.get_registry_version()
            cached = self._cache.get(surface_form, registry_version)
            if cached is not None:
                return cached

        # --- Execute strategy sequence ------------------------------------
        result = self._execute_strategies(surface_form, entity_type, paper_id, now)

        # --- Write audit record (non-blocking) ----------------------------
        self._write_audit_record(result)

        # --- Update metrics -----------------------------------------------
        self._update_metrics(result)

        # --- Store in cache -----------------------------------------------
        if self._cache is not None and self._registry is not None:
            registry_version = self._registry.get_registry_version()
            try:
                self._cache.put(surface_form, result, registry_version)
            except Exception as exc:
                logger.error(
                    "ResolutionPipeline: cache.put() failed for '%s': %s",
                    surface_form,
                    exc,
                )

        return result

    def normalize(
        self,
        surface_form: str,
        entity_type: str,
        paper_id: str = "unknown",
    ) -> NormalizationResult:
        """
        Drop-in replacement for Spec 1 Entity_Normalizer.normalize().

        Returns NormalizationResult(canonical_id, grounded).

        When shadow_mode=True, delegates to normalize_shadow_mode() which runs
        both Spec 1 and Spec 2 normalizers, logs ShadowModeDiscrepancy when
        results differ, and returns the Spec 1 result.

        Requirements: 14.1, 14.3, 14.4, 14.5
        """
        if self._shadow_mode:
            return self.normalize_shadow_mode(surface_form, entity_type, paper_id)

        result = self.resolve(surface_form, entity_type, paper_id)
        return NormalizationResult(
            canonical_id=result.canonical_id,
            grounded=result.grounded,
        )

    def normalize_shadow_mode(
        self,
        surface_form: str,
        entity_type: str,
        paper_id: str = "unknown",
    ) -> NormalizationResult:
        """
        Run both Spec 1 and Spec 2 normalizers in parallel.

        Logs a ShadowModeDiscrepancy when the two normalizers produce different
        results (different canonical_id or different grounded flag).

        Always returns the Spec 1 result so that existing graph construction
        behaviour is unchanged during the shadow period.

        If no Spec 1 normalizer is configured, falls back to the Spec 2 result
        and logs a warning (full wiring is completed in task 16.1).

        Requirements: 14.5, 14.6
        """
        now = datetime.now(timezone.utc)

        # --- Spec 2 result (this pipeline) --------------------------------
        spec2_result = self.resolve(surface_form, entity_type, paper_id)
        spec2_norm = NormalizationResult(
            canonical_id=spec2_result.canonical_id,
            grounded=spec2_result.grounded,
        )

        # --- Spec 1 result ------------------------------------------------
        if self._spec1_normalizer is None:
            # Spec 1 normalizer not yet wired (task 16.1); return Spec 2 result
            logger.warning(
                "ResolutionPipeline: shadow mode enabled but no Spec 1 normalizer "
                "configured; returning Spec 2 result for surface_form=%r",
                surface_form,
            )
            return spec2_norm

        try:
            spec1_norm: NormalizationResult = self._spec1_normalizer.normalize(
                surface_form, entity_type
            )
        except Exception as exc:
            logger.error(
                "ResolutionPipeline: Spec 1 normalizer raised an exception for "
                "surface_form=%r: %s; returning Spec 2 result",
                surface_form,
                exc,
            )
            return spec2_norm

        # --- Compare and log discrepancy if results differ ----------------
        if (
            spec1_norm.canonical_id != spec2_norm.canonical_id
            or spec1_norm.grounded != spec2_norm.grounded
        ):
            discrepancy = ShadowModeDiscrepancy(
                surface_form=surface_form,
                entity_type=entity_type,
                paper_id=paper_id,
                spec1_canonical_id=spec1_norm.canonical_id,
                spec1_grounded=spec1_norm.grounded,
                spec2_canonical_id=spec2_norm.canonical_id,
                spec2_grounded=spec2_norm.grounded,
                timestamp=now,
            )
            logger.info(
                "ResolutionPipeline: shadow mode discrepancy for surface_form=%r "
                "entity_type=%r paper_id=%r | spec1=(canonical_id=%r, grounded=%s) "
                "spec2=(canonical_id=%r, grounded=%s)",
                surface_form,
                entity_type,
                paper_id,
                discrepancy.spec1_canonical_id,
                discrepancy.spec1_grounded,
                discrepancy.spec2_canonical_id,
                discrepancy.spec2_grounded,
            )

        # Always return Spec 1 result (shadow mode does not change behaviour)
        return spec1_norm

    def batch_resolve(
        self,
        forms: List[Tuple[str, str, str]],  # (surface_form, entity_type, paper_id)
    ) -> List[ResolutionResult]:
        """
        Resolve a batch of 1–100,000 surface forms.

        Returns results in the same order as input.

        Requirements: 8.1
        """
        return [
            self.resolve(surface_form, entity_type, paper_id)
            for surface_form, entity_type, paper_id in forms
        ]

    # ------------------------------------------------------------------
    # Shadow mode
    # ------------------------------------------------------------------

    def enable_shadow_mode(self) -> None:
        """Enable shadow mode (run alongside Spec 1 normalizer)."""
        self._shadow_mode = True

    def disable_shadow_mode(self) -> None:
        """Disable shadow mode."""
        self._shadow_mode = False

    # ------------------------------------------------------------------
    # Core strategy execution
    # ------------------------------------------------------------------

    def _execute_strategies(
        self,
        surface_form: str,
        entity_type: str,
        paper_id: str,
        now: datetime,
    ) -> ResolutionResult:
        """
        Execute the seven-strategy sequence and return a ResolutionResult.

        Collects all candidates from all strategies into conflict_set.
        Accepts the first strategy that produces a candidate with
        grounding_confidence >= 0.5 (Requirement 1.2).

        Requirements: 1.1, 1.2, 1.3, 1.4, 1.5
        """
        # Normalise the surface form before any comparison (Requirement 2.1)
        normalised = normalize_surface_form(surface_form)

        # Track which strategies produced candidates (for high_conflict flag)
        strategies_with_candidates: List[str] = []
        # All candidates collected across all strategies
        all_candidates: List[CandidateScore] = []

        # ----------------------------------------------------------------
        # Strategy 1: Manual Override lookup
        # ----------------------------------------------------------------
        override_candidate = self._try_manual_override(surface_form, entity_type)
        if override_candidate is not None:
            all_candidates.append(override_candidate)
            strategies_with_candidates.append("manual_override")
            if override_candidate.grounding_confidence >= _CONFIDENCE_THRESHOLD:
                return self._build_result(
                    surface_form=surface_form,
                    entity_type=entity_type,
                    paper_id=paper_id,
                    now=now,
                    winner=override_candidate,
                    all_candidates=all_candidates,
                    strategies_with_candidates=strategies_with_candidates,
                    hierarchy_level=None,
                )

        # ----------------------------------------------------------------
        # Strategies 2–6 (and abbreviation re-entry): run on original form
        # then optionally on expanded forms
        # ----------------------------------------------------------------
        winner, all_candidates, strategies_with_candidates, hierarchy_level = (
            self._run_strategies_2_to_6(
                surface_form=surface_form,
                normalised=normalised,
                entity_type=entity_type,
                all_candidates=all_candidates,
                strategies_with_candidates=strategies_with_candidates,
                allow_abbreviation_reentry=True,
            )
        )

        if winner is not None:
            return self._build_result(
                surface_form=surface_form,
                entity_type=entity_type,
                paper_id=paper_id,
                now=now,
                winner=winner,
                all_candidates=all_candidates,
                strategies_with_candidates=strategies_with_candidates,
                hierarchy_level=hierarchy_level,
            )

        # ----------------------------------------------------------------
        # Strategy 7: OntologyTraverser hierarchy search
        # ----------------------------------------------------------------
        ontology_candidates = self._try_ontology(surface_form, entity_type)
        if ontology_candidates:
            strategies_with_candidates.append("ontology")
            all_candidates.extend(ontology_candidates)
            # Filter to those meeting the threshold
            qualifying = [
                c for c in ontology_candidates
                if c.grounding_confidence >= _CONFIDENCE_THRESHOLD
            ]
            if qualifying:
                winner = self._ranking_function.rank(qualifying)
                # Find hierarchy_level from the OntologyCandidate
                hier_level = self._get_hierarchy_level_from_candidates(
                    winner.canonical_id, ontology_candidates
                )
                return self._build_result(
                    surface_form=surface_form,
                    entity_type=entity_type,
                    paper_id=paper_id,
                    now=now,
                    winner=winner,
                    all_candidates=all_candidates,
                    strategies_with_candidates=strategies_with_candidates,
                    hierarchy_level=hier_level,
                )

        # ----------------------------------------------------------------
        # All strategies failed — create UnresolvedEntity
        # ----------------------------------------------------------------
        return self._build_unresolved_result(
            surface_form=surface_form,
            entity_type=entity_type,
            paper_id=paper_id,
            now=now,
            all_candidates=all_candidates,
            strategies_with_candidates=strategies_with_candidates,
        )

    def _run_strategies_2_to_6(
        self,
        surface_form: str,
        normalised: str,
        entity_type: str,
        all_candidates: List[CandidateScore],
        strategies_with_candidates: List[str],
        allow_abbreviation_reentry: bool,
    ) -> Tuple[Optional[CandidateScore], List[CandidateScore], List[str], Optional[int]]:
        """
        Run strategies 2–6 on the given surface form.

        Returns (winner, all_candidates, strategies_with_candidates, hierarchy_level).
        winner is None if no strategy produced a qualifying match.

        Handles abbreviation re-entry (at most once) by recursing with
        allow_abbreviation_reentry=False.

        Requirements: 1.1, 1.2, 1.5
        """
        # ----------------------------------------------------------------
        # Strategy 2: Exact match against CanonicalRegistry
        # ----------------------------------------------------------------
        exact_candidate = self._try_exact_match(surface_form)
        if exact_candidate is not None:
            all_candidates.append(exact_candidate)
            if "exact" not in strategies_with_candidates:
                strategies_with_candidates.append("exact")
            if exact_candidate.grounding_confidence >= _CONFIDENCE_THRESHOLD:
                return exact_candidate, all_candidates, strategies_with_candidates, None

        # ----------------------------------------------------------------
        # Strategy 3: Normalized match
        # ----------------------------------------------------------------
        norm_candidate = self._try_normalized_match(normalised)
        if norm_candidate is not None:
            all_candidates.append(norm_candidate)
            if "normalized" not in strategies_with_candidates:
                strategies_with_candidates.append("normalized")
            if norm_candidate.grounding_confidence >= _CONFIDENCE_THRESHOLD:
                return norm_candidate, all_candidates, strategies_with_candidates, None

        # ----------------------------------------------------------------
        # Strategy 4: AbbreviationExpander lookup + re-entry
        # ----------------------------------------------------------------
        if allow_abbreviation_reentry and self._abbreviation_expander is not None:
            expansions = []
            try:
                expansions = self._abbreviation_expander.expand(surface_form)
            except Exception as exc:
                logger.warning(
                    "ResolutionPipeline: abbreviation_expander.expand() failed for '%s': %s",
                    surface_form,
                    exc,
                )

            if expansions:
                # Try each expansion in lexicographic order through steps 2–6
                # (expansions are already sorted lexicographically by AbbreviationExpander)
                for expansion in expansions:
                    expanded_form = expansion.expanded_form
                    expanded_normalised = normalize_surface_form(expanded_form)

                    # Re-enter from step 2 with expanded form (at most once)
                    (
                        re_winner,
                        all_candidates,
                        strategies_with_candidates,
                        re_hier_level,
                    ) = self._run_strategies_2_to_6(
                        surface_form=expanded_form,
                        normalised=expanded_normalised,
                        entity_type=entity_type,
                        all_candidates=all_candidates,
                        strategies_with_candidates=strategies_with_candidates,
                        allow_abbreviation_reentry=False,  # at most once
                    )
                    if re_winner is not None:
                        # Tag the winner as coming from abbreviation strategy
                        abbr_winner = CandidateScore(
                            canonical_id=re_winner.canonical_id,
                            strategy="abbreviation",
                            grounding_confidence=expansion.confidence,
                            composite_score=0.80 * expansion.confidence,
                        )
                        # Replace the re_winner in all_candidates with abbr_winner
                        # (keep the original re_winner too for full audit trail)
                        all_candidates.append(abbr_winner)
                        if "abbreviation" not in strategies_with_candidates:
                            strategies_with_candidates.append("abbreviation")
                        if abbr_winner.grounding_confidence >= _CONFIDENCE_THRESHOLD:
                            return abbr_winner, all_candidates, strategies_with_candidates, re_hier_level

        # ----------------------------------------------------------------
        # Strategy 5: SynonymIndex lookup
        # ----------------------------------------------------------------
        synonym_candidate = self._try_synonym_lookup(surface_form)
        if synonym_candidate is not None:
            all_candidates.append(synonym_candidate)
            if "synonym" not in strategies_with_candidates:
                strategies_with_candidates.append("synonym")
            if synonym_candidate.grounding_confidence >= _CONFIDENCE_THRESHOLD:
                return synonym_candidate, all_candidates, strategies_with_candidates, None

        # ----------------------------------------------------------------
        # Strategy 6: FuzzyMatcher (skip if < 4 code points)
        # ----------------------------------------------------------------
        fuzzy_candidates = self._try_fuzzy_match(surface_form, entity_type)
        if fuzzy_candidates:
            all_candidates.extend(fuzzy_candidates)
            if "fuzzy" not in strategies_with_candidates:
                strategies_with_candidates.append("fuzzy")
            qualifying = [
                c for c in fuzzy_candidates
                if c.grounding_confidence >= _CONFIDENCE_THRESHOLD
            ]
            if qualifying:
                winner = self._ranking_function.rank(qualifying)
                return winner, all_candidates, strategies_with_candidates, None

        return None, all_candidates, strategies_with_candidates, None

    # ------------------------------------------------------------------
    # Individual strategy helpers
    # ------------------------------------------------------------------

    def _try_manual_override(
        self, surface_form: str, entity_type: str
    ) -> Optional[CandidateScore]:
        """
        Strategy 1: Check for a Manual Override.

        Returns a CandidateScore with confidence=1.0 if an override exists,
        None otherwise.

        Requirements: 9.1, 9.2, 9.4
        """
        if self._override_manager is None:
            return None
        try:
            override = self._override_manager.get_override(surface_form)
            if override is not None:
                return CandidateScore(
                    canonical_id=override.canonical_id,
                    strategy="manual_override",
                    grounding_confidence=1.0,
                    composite_score=1.0,  # 1.00 * 1.0
                )
        except Exception as exc:
            logger.warning(
                "ResolutionPipeline: override_manager.get_override() failed for '%s': %s",
                surface_form,
                exc,
            )
        return None

    def _try_exact_match(self, surface_form: str) -> Optional[CandidateScore]:
        """
        Strategy 2: Exact match against CanonicalRegistry.

        Looks up the surface form as-is (the registry does NFC+lowercase internally).
        Returns a CandidateScore with confidence=1.0 if found, None otherwise.

        Requirements: 1.1
        """
        if self._registry is None:
            return None
        try:
            record = self._registry.lookup_by_surface_form(surface_form)
            if record is not None:
                return CandidateScore(
                    canonical_id=record.canonical_id,
                    strategy="exact",
                    grounding_confidence=1.0,
                    composite_score=0.95,  # 0.95 * 1.0
                )
        except Exception as exc:
            logger.warning(
                "ResolutionPipeline: registry.lookup_by_surface_form() failed for '%s': %s",
                surface_form,
                exc,
            )
        return None

    def _try_normalized_match(self, normalised: str) -> Optional[CandidateScore]:
        """
        Strategy 3: Normalized match (case-fold, strip punctuation, collapse whitespace).

        Looks up the pre-normalised form in the registry.
        Returns a CandidateScore with confidence=1.0 if found, None otherwise.

        Requirements: 1.1
        """
        if self._registry is None or not normalised:
            return None
        try:
            record = self._registry.lookup_by_surface_form(normalised)
            if record is not None:
                return CandidateScore(
                    canonical_id=record.canonical_id,
                    strategy="normalized",
                    grounding_confidence=1.0,
                    composite_score=0.85,  # 0.85 * 1.0
                )
        except Exception as exc:
            logger.warning(
                "ResolutionPipeline: registry.lookup_by_surface_form() (normalized) "
                "failed for '%s': %s",
                normalised,
                exc,
            )
        return None

    def _try_synonym_lookup(self, surface_form: str) -> Optional[CandidateScore]:
        """
        Strategy 5: SynonymIndex lookup.

        Returns a CandidateScore with confidence=1.0 if found, None otherwise.

        Requirements: 1.1
        """
        if self._synonym_index is None:
            return None
        try:
            canonical_id = self._synonym_index.lookup(surface_form)
            if canonical_id is not None:
                return CandidateScore(
                    canonical_id=canonical_id,
                    strategy="synonym",
                    grounding_confidence=1.0,
                    composite_score=0.75,  # 0.75 * 1.0
                )
        except Exception as exc:
            logger.warning(
                "ResolutionPipeline: synonym_index.lookup() failed for '%s': %s",
                surface_form,
                exc,
            )
        return None

    def _try_fuzzy_match(
        self, surface_form: str, entity_type: str
    ) -> List[CandidateScore]:
        """
        Strategy 6: FuzzyMatcher (edit distance ≤ 2, skip if < 4 code points).

        Returns a list of CandidateScore objects (may be empty).

        Requirements: 12.1, 12.5
        """
        if self._fuzzy_matcher is None or self._registry is None:
            return []
        try:
            fuzzy_results = self._fuzzy_matcher.match(
                surface_form, entity_type, self._registry
            )
            candidates = []
            for fc in fuzzy_results:
                candidates.append(
                    CandidateScore(
                        canonical_id=fc.canonical_id,
                        strategy="fuzzy",
                        grounding_confidence=fc.grounding_confidence,
                        composite_score=0.60 * fc.grounding_confidence,
                    )
                )
            return candidates
        except Exception as exc:
            logger.warning(
                "ResolutionPipeline: fuzzy_matcher.match() failed for '%s': %s",
                surface_form,
                exc,
            )
        return []

    def _try_ontology(
        self, surface_form: str, entity_type: str
    ) -> List[CandidateScore]:
        """
        Strategy 7: OntologyTraverser hierarchy search.

        Returns a list of CandidateScore objects (may be empty).
        Also stores hierarchy_level on the candidates for later use.

        Requirements: 13.1, 13.4, 13.5
        """
        if self._ontology_traverser is None or self._registry is None:
            return []
        try:
            ontology_results = self._ontology_traverser.traverse(
                surface_form, entity_type, self._registry
            )
            candidates = []
            for oc in ontology_results:
                candidates.append(
                    CandidateScore(
                        canonical_id=oc.canonical_id,
                        strategy="ontology",
                        grounding_confidence=oc.grounding_confidence,
                        composite_score=0.50 * oc.grounding_confidence,
                    )
                )
            return candidates
        except Exception as exc:
            logger.warning(
                "ResolutionPipeline: ontology_traverser.traverse() failed for '%s': %s",
                surface_form,
                exc,
            )
        return []

    # ------------------------------------------------------------------
    # Result builders
    # ------------------------------------------------------------------

    def _build_result(
        self,
        surface_form: str,
        entity_type: str,
        paper_id: str,
        now: datetime,
        winner: CandidateScore,
        all_candidates: List[CandidateScore],
        strategies_with_candidates: List[str],
        hierarchy_level: Optional[int],
    ) -> ResolutionResult:
        """
        Build a successful ResolutionResult from the winning candidate.

        Sets high_conflict=True when ≥3 strategies produced candidates.
        Sets hierarchy_match=True and hierarchy_level when OntologyTraverser wins.

        Requirements: 1.2, 4.4, 4.5, 13.4, 13.5
        """
        high_conflict = len(strategies_with_candidates) >= 3
        hierarchy_match = winner.strategy == "ontology"

        # Score all candidates for the conflict_set
        scored_candidates = self._ranking_function.score_all(all_candidates)

        # Determine curator_override if manual override won
        curator_override: Optional[str] = None
        if winner.strategy == "manual_override" and self._override_manager is not None:
            try:
                override = self._override_manager.get_override(surface_form)
                if override is not None:
                    curator_override = override.curator_id
            except Exception:
                pass

        return ResolutionResult(
            surface_form=surface_form,
            entity_type=entity_type,
            canonical_id=winner.canonical_id,
            grounded=True,
            winning_strategy=winner.strategy,
            grounding_confidence=winner.grounding_confidence,
            conflict_set=scored_candidates,
            paper_id=paper_id,
            timestamp=now,
            high_conflict=high_conflict,
            hierarchy_match=hierarchy_match,
            hierarchy_level=hierarchy_level if hierarchy_match else None,
        )

    def _build_unresolved_result(
        self,
        surface_form: str,
        entity_type: str,
        paper_id: str,
        now: datetime,
        all_candidates: List[CandidateScore],
        strategies_with_candidates: List[str],
    ) -> ResolutionResult:
        """
        Build an unresolved ResolutionResult when all strategies fail.

        Creates an UnresolvedEntity record and returns a ResolutionResult
        with grounded=False and winning_strategy="none".

        Requirements: 1.3, 14.4
        """
        # Create UnresolvedEntity record
        unresolved = UnresolvedEntity(
            surface_form=surface_form,
            entity_type=entity_type,
            paper_id=paper_id,
            timestamp=now,
            local_id=f"UNRESOLVED-{uuid.uuid4().hex[:8].upper()}",
        )
        logger.info(
            "ResolutionPipeline: unresolved entity created for '%s' (entity_type=%s, "
            "paper_id=%s, local_id=%s)",
            surface_form,
            entity_type,
            paper_id,
            unresolved.local_id,
        )

        high_conflict = len(strategies_with_candidates) >= 3
        scored_candidates = (
            self._ranking_function.score_all(all_candidates)
            if all_candidates
            else []
        )

        return ResolutionResult(
            surface_form=surface_form,
            entity_type=entity_type,
            canonical_id=None,
            grounded=False,
            winning_strategy="none",
            grounding_confidence=0.0,
            conflict_set=scored_candidates,
            paper_id=paper_id,
            timestamp=now,
            high_conflict=high_conflict,
            hierarchy_match=False,
            hierarchy_level=None,
        )

    # ------------------------------------------------------------------
    # Audit, metrics, and cache helpers
    # ------------------------------------------------------------------

    def _write_audit_record(self, result: ResolutionResult) -> None:
        """
        Write a ResolutionRecord to the audit store (non-blocking).

        Logs failure but does not raise (Requirement 7.5).
        """
        if self._audit_store is None:
            return
        try:
            record = ResolutionRecord(
                surface_form=result.surface_form,
                entity_type=result.entity_type,
                timestamp=result.timestamp,
                winning_strategy=result.winning_strategy,
                canonical_id=result.canonical_id,
                grounding_confidence=result.grounding_confidence,
                conflict_set=result.conflict_set,
                paper_id=result.paper_id,
                high_conflict=result.high_conflict,
                hierarchy_match=result.hierarchy_match,
                hierarchy_level=result.hierarchy_level,
                curator_override=None,  # populated in _build_result for manual_override
            )
            # For manual_override, extract curator_id from the result's conflict_set
            if result.winning_strategy == "manual_override" and self._override_manager is not None:
                try:
                    override = self._override_manager.get_override(result.surface_form)
                    if override is not None:
                        record = record.model_copy(
                            update={"curator_override": override.curator_id}
                        )
                except Exception:
                    pass

            success = self._audit_store.write(record)
            if not success:
                logger.error(
                    "ResolutionPipeline: audit_store.write() returned False for "
                    "surface_form=%r paper_id=%r",
                    result.surface_form,
                    result.paper_id,
                )
        except Exception as exc:
            logger.error(
                "ResolutionPipeline: _write_audit_record() failed for "
                "surface_form=%r paper_id=%r: %s",
                result.surface_form,
                result.paper_id,
                exc,
            )

    def _update_metrics(self, result: ResolutionResult) -> None:
        """
        Update ResolutionMetrics with the resolution result (non-blocking).
        """
        if self._metrics is None:
            return
        try:
            self._metrics.record_resolution(result)
        except Exception as exc:
            logger.error(
                "ResolutionPipeline: metrics.record_resolution() failed: %s", exc
            )

    def _get_hierarchy_level_from_candidates(
        self,
        canonical_id: str,
        ontology_candidates: List[CandidateScore],
    ) -> Optional[int]:
        """
        Derive the hierarchy_level for an ontology winner from the raw
        OntologyCandidate confidence value.

        confidence = 0.50 - (level - 1) * 0.10
        => level = round((0.50 - confidence) / 0.10) + 1
        """
        for candidate in ontology_candidates:
            if candidate.canonical_id == canonical_id:
                conf = candidate.grounding_confidence
                # Invert: level = round((0.50 - conf) / 0.10) + 1
                level = round((0.50 - conf) / 0.10) + 1
                if 1 <= level <= 3:
                    return level
        return None
