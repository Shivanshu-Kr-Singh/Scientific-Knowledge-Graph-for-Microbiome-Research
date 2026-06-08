"""
graph/entity_normalizer.py — Generalized ontology registry architecture.

Implements a routing-table-driven entity normalizer with:
  - Persistent SQLite grounding cache (never calls the same API twice)
  - NCBI Entrez grounding for taxon, disease, gene, treatment
  - UniProt REST grounding for protein
  - EMBL-EBI OLS4 grounding for metabolite, pathway, body_site, cell_type,
    immune_cell, phenotype, drug, method
  - OLS cross-search fallback for unknown entity types
  - LLM fallback (temperature=0, deterministic) as last resort
  - Failure logging to SQLite (unchanged schema)
  - Backward-compatible normalize_taxon() / normalize_disease() wrappers

Concern 5 — generalized ontology registry.
"""

import io
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import requests
import yaml
from loguru import logger

try:
    from Bio import Entrez

    ENTREZ_AVAILABLE = True
except ImportError:
    ENTREZ_AVAILABLE = False

# ─── Paths ────────────────────────────────────────────────────────────────────

SCHEMA_DIR = Path(__file__).parent / "schemas"
DB_PATH = Path(__file__).parent / "entity_normalization.db"
CACHE_DB_PATH = Path(__file__).parent / "grounding_cache.db"

# ─── NCBI e-mail ──────────────────────────────────────────────────────────────

ENTREZ_EMAIL = os.getenv("NCBI_EMAIL", "research@example.com")
ENTREZ_API_KEY = os.getenv("NCBI_API_KEY", "")
if ENTREZ_AVAILABLE:
    Entrez.email = ENTREZ_EMAIL
    if ENTREZ_API_KEY:
        Entrez.api_key = ENTREZ_API_KEY

# NCBI rate-limit: 3 req/sec without key, 10 with key.
# 0.34 s gap keeps us safely under the no-key limit;
# 0.12 s is sufficient with an API key.
_NCBI_DELAY = 0.12 if ENTREZ_API_KEY else 0.34

# ─── Confidence levels ────────────────────────────────────────────────────────

CONFIDENCE_AUTHORITATIVE = 1.0   # Exact match in authoritative ontology
CONFIDENCE_FUZZY = 0.8           # Fuzzy match in authoritative ontology
CONFIDENCE_LLM = 0.6             # LLM-suggested grounding
CONFIDENCE_UNGROUNDED = 0.0      # Failed all attempts

# ─── Ontology routing table ───────────────────────────────────────────────────

ONTOLOGY_ROUTING: Dict[str, Dict[str, Any]] = {
    "taxon": {
        "ontologies": ["ncbi_taxonomy"],
        "api": "ncbi_taxonomy",
        "ncbi_db": "taxonomy",
    },
    "disease": {
        "ontologies": ["mesh", "omim"],
        "api": "ncbi_mesh",
        "ncbi_db": "mesh",
    },
    "gene": {
        "ontologies": ["ncbi_gene"],
        "api": "ncbi_gene",
        "ncbi_db": "gene",
    },
    "protein": {
        "ontologies": ["uniprot"],
        "api": "uniprot",
    },
    "metabolite": {
        "ontologies": ["chebi"],
        "api": "ols",
        "ols_ontologies": "chebi",
    },
    "pathway": {
        "ontologies": ["kegg", "reactome"],
        "api": "ols",
        "ols_ontologies": "go,pw",
    },
    "body_site": {
        "ontologies": ["uberon"],
        "api": "ols",
        "ols_ontologies": "uberon,bto",
    },
    "cell_type": {
        "ontologies": ["cl"],
        "api": "ols",
        "ols_ontologies": "cl",
    },
    "immune_cell": {
        "ontologies": ["cl"],
        "api": "ols",
        "ols_ontologies": "cl",
    },
    "phenotype": {
        "ontologies": ["hp"],
        "api": "ols",
        "ols_ontologies": "hp",
    },
    "drug": {
        "ontologies": ["chebi", "chembl"],
        "api": "ols",
        "ols_ontologies": "chebi,obi",
    },
    "treatment": {
        "ontologies": ["mesh"],
        "api": "ncbi_mesh",
        "ncbi_db": "mesh",
    },
    "method": {
        "ontologies": ["obi"],
        "api": "ols",
        "ols_ontologies": "obi,efo",
    },
    # Default for unknown types
    "_default": {
        "ontologies": ["ols_all"],
        "api": "ols_cross",
    },
}




# ─── EntityNormalizer ─────────────────────────────────────────────────────────


class EntityNormalizer:
    """
    Generalized ontology registry normalizer.

    Single entry point: normalize(entity_text, entity_type) → dict
    Backward-compatible wrappers: normalize_taxon(), normalize_disease()
    """

    def __init__(self) -> None:
        # Load YAML abbreviation maps (fast local lookup before any API call)
        self.taxa:         dict = self._load_yaml("taxonomy_map.yaml")
        self.disease_map:  dict = self._load_yaml("disease_map.yaml")
        self.method_map:   dict = self._load_yaml("method_map.yaml")
        self.metabolite_map: dict = self._load_yaml("metabolite_map.yaml")
        self.gene_map:     dict = self._load_yaml("gene_map.yaml")
        self.treatment_map: dict = self._load_yaml("treatment_map.yaml")
        self.body_site_map: dict = self._load_yaml("body_site_map.yaml")
        self.dataset_map:  dict = self._load_yaml("dataset_map.yaml")

        # Map entity types to their YAML abbreviation maps
        self._yaml_maps = {
            "taxon":            self.taxa,
            "disease":          self.disease_map,
            "method":           self.method_map,
            "metabolite":       self.metabolite_map,
            "gene":             self.gene_map,
            "treatment":        self.treatment_map,
            "body_site":        self.body_site_map,
            "dataset":          self.dataset_map,
        }

        # Failure log DB (existing schema, unchanged)
        self._init_failure_log_db()

        # Grounding cache DB (new)
        self._init_grounding_cache()

        # LLM grounder — loaded lazily on first use
        self._llm_grounder = None

    # ── YAML loading ──────────────────────────────────────────────────────────

    def _load_yaml(self, fname: str) -> Dict[str, list]:
        path = SCHEMA_DIR / fname
        if not path.exists():
            return {}
        try:
            with open(path) as fh:
                data = yaml.safe_load(fh)
            return data or {}
        except yaml.YAMLError as e:
            logger.error(f"[EntityNormalizer] YAML parse error in {fname}: {e}")
            return {}
        except Exception as e:
            logger.warning(f"[EntityNormalizer] Could not load {fname}: {e}")
            return {}

    # ── Failure log DB (unchanged schema) ────────────────────────────────────

    def _init_failure_log_db(self) -> None:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS entity_normalization_failures (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_text TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                failure_reason TEXT NOT NULL,
                attempted_matches TEXT,
                timestamp TEXT NOT NULL,
                grounded BOOLEAN DEFAULT 0
            )
            """
        )
        conn.commit()
        conn.close()

    def _log_failure(
        self,
        entity_text: str,
        entity_type: str,
        failure_reason: str,
        attempted_matches: str = "",
    ) -> None:
        try:
            conn = sqlite3.connect(DB_PATH)
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO entity_normalization_failures
                    (entity_text, entity_type, failure_reason,
                     attempted_matches, timestamp, grounded)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    entity_text,
                    entity_type,
                    failure_reason,
                    attempted_matches,
                    datetime.now(timezone.utc).isoformat(),
                    False,
                ),
            )
            conn.commit()
            conn.close()
        except Exception as exc:  # noqa: BLE001
            logger.error("EntityNormalizer: could not write failure log — {}", exc)

    # ── Grounding cache DB ────────────────────────────────────────────────────

    def _init_grounding_cache(self) -> None:
        conn = sqlite3.connect(CACHE_DB_PATH)
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS grounding_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_text TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                canonical_name TEXT NOT NULL,
                ontology_id TEXT,
                ontology_name TEXT,
                confidence REAL NOT NULL,
                source TEXT NOT NULL,
                grounded BOOLEAN NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(entity_text, entity_type)
            )
            """
        )
        conn.commit()
        conn.close()

    def _cache_lookup(
        self, entity_text: str, entity_type: str
    ) -> Optional[Dict[str, Any]]:
        """Return cached grounding dict or None."""
        try:
            conn = sqlite3.connect(CACHE_DB_PATH)
            cur = conn.cursor()
            cur.execute(
                """
                SELECT canonical_name, ontology_id, ontology_name,
                       confidence, source, grounded
                FROM grounding_cache
                WHERE entity_text = ? AND entity_type = ?
                """,
                (entity_text.lower(), entity_type),
            )
            row = cur.fetchone()
            conn.close()
            if row is None:
                return None
            canonical_name, ontology_id, ontology_name, confidence, source, grounded = row
            return {
                "id": ontology_id or f"ungrounded:{entity_text.lower()}",
                "name": entity_text,
                "canonical_name": canonical_name,
                "ontology": ontology_name,
                "confidence": confidence,
                "grounded": bool(grounded),
                "source": source,
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("EntityNormalizer: cache lookup failed — {}", exc)
            return None

    def _cache_store(
        self, entity_text: str, entity_type: str, result: Dict[str, Any]
    ) -> None:
        """Insert or replace a grounding result in the cache."""
        try:
            conn = sqlite3.connect(CACHE_DB_PATH)
            cur = conn.cursor()
            cur.execute(
                """
                INSERT OR REPLACE INTO grounding_cache
                    (entity_text, entity_type, canonical_name, ontology_id,
                     ontology_name, confidence, source, grounded, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entity_text.lower(),
                    entity_type,
                    result.get("canonical_name", entity_text),
                    result.get("id"),
                    result.get("ontology"),
                    result.get("confidence", CONFIDENCE_UNGROUNDED),
                    result.get("source", "unknown"),
                    result.get("grounded", False),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            conn.commit()
            conn.close()
        except Exception as exc:  # noqa: BLE001
            logger.warning("EntityNormalizer: cache store failed — {}", exc)

    # ── Lazy LLM grounder ─────────────────────────────────────────────────────

    def _get_llm_grounder(self):
        if self._llm_grounder is None:
            try:
                from semantic.llm_grounder import LLMGrounder

                self._llm_grounder = LLMGrounder()
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "EntityNormalizer: could not load LLMGrounder — {}", exc
                )
        return self._llm_grounder

    # ── YAML abbreviation lookup ──────────────────────────────────────────────

    def _yaml_lookup(
        self, entity_text: str, entity_type: str
    ) -> Optional[str]:
        """
        Check local YAML maps for common abbreviations and synonyms.
        Returns canonical key string or None.
        Covers all 8 entity types that have YAML maps.
        """
        text_lower = entity_text.lower().strip()
        mapping = self._yaml_maps.get(entity_type)
        if not mapping:
            return None

        for canonical, variants in mapping.items():
            if text_lower in [v.lower() for v in variants]:
                return canonical

        return None

    # ── NCBI grounding ────────────────────────────────────────────────────────

    def _ground_via_ncbi(
        self, entity_text: str, ncbi_db: str
    ) -> Optional[Dict[str, Any]]:
        """
        Ground entity_text against an NCBI database via Entrez.
        Respects the 3 req/sec rate limit with a 0.34 s inter-call delay.
        Returns a grounding dict or None on any failure.
        """
        if not ENTREZ_AVAILABLE:
            logger.warning(
                "EntityNormalizer: Biopython not available; skipping NCBI lookup"
            )
            return None

        try:
            handle = Entrez.esearch(db=ncbi_db, term=entity_text, retmax=1)
            raw = handle.read()
            handle.close()
            if isinstance(raw, str):
                raw = raw.encode("utf-8")
            # Guard: NCBI occasionally returns an HTML error page (rate limit, 5xx)
            # instead of XML.  Detect this before passing to Entrez.read().
            raw_preview = raw[:200].lstrip()
            if not raw_preview.startswith(b"<?xml") and not raw_preview.startswith(b"<"):
                logger.debug(
                    "EntityNormalizer: NCBI esearch returned non-XML response for {!r} — skipping",
                    entity_text,
                )
                return None
            record = Entrez.read(io.BytesIO(raw))
            time.sleep(_NCBI_DELAY)

            id_list = record.get("IdList", [])
            if not id_list:
                return None

            entity_id = id_list[0]

            handle = Entrez.efetch(
                db=ncbi_db, id=entity_id, retmode="xml"
            )
            raw = handle.read()
            handle.close()
            if isinstance(raw, str):
                raw = raw.encode("utf-8")
            # Same guard for efetch response
            raw_preview = raw[:200].lstrip()
            if not raw_preview.startswith(b"<?xml") and not raw_preview.startswith(b"<"):
                logger.debug(
                    "EntityNormalizer: NCBI efetch returned non-XML response for {!r} (db={!r}) — skipping",
                    entity_text,
                    ncbi_db,
                )
                return None
            records = Entrez.read(io.BytesIO(raw))
            time.sleep(_NCBI_DELAY)

            if not records:
                return None

            rec = records[0]

            # ── taxonomy ──────────────────────────────────────────────────────
            if ncbi_db == "taxonomy":
                return {
                    "id": f"ncbi:{entity_id}",
                    "canonical_name": rec.get("ScientificName", entity_text),
                    "ontology": "NCBI Taxonomy",
                    "confidence": CONFIDENCE_AUTHORITATIVE,
                    "grounded": True,
                    "source": "ncbi",
                }

            # ── mesh ──────────────────────────────────────────────────────────
            if ncbi_db == "mesh":
                descriptor = rec.get("DescriptorName", {})
                canonical = (
                    descriptor.get("String", entity_text)
                    if isinstance(descriptor, dict)
                    else entity_text
                )
                return {
                    "id": f"mesh:{entity_id}",
                    "canonical_name": canonical,
                    "ontology": "MeSH",
                    "confidence": CONFIDENCE_AUTHORITATIVE,
                    "grounded": True,
                    "source": "ncbi",
                }

            # ── gene ──────────────────────────────────────────────────────────
            if ncbi_db == "gene":
                # Gene records are nested; extract the official symbol
                entrez_gene = rec.get("Entrezgene_gene", {})
                gene_ref = entrez_gene.get("Gene-ref", {})
                symbol = gene_ref.get("Gene-ref_locus", entity_text)
                return {
                    "id": f"ncbi_gene:{entity_id}",
                    "canonical_name": symbol,
                    "ontology": "NCBI Gene",
                    "confidence": CONFIDENCE_AUTHORITATIVE,
                    "grounded": True,
                    "source": "ncbi",
                }

            # ── generic fallback for other NCBI dbs ───────────────────────────
            return {
                "id": f"{ncbi_db}:{entity_id}",
                "canonical_name": entity_text,
                "ontology": ncbi_db.upper(),
                "confidence": CONFIDENCE_AUTHORITATIVE,
                "grounded": True,
                "source": "ncbi",
            }

        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "EntityNormalizer: NCBI lookup failed for {!r} in db={!r} — {}",
                entity_text,
                ncbi_db,
                exc,
            )
            return None

    # ── UniProt grounding ─────────────────────────────────────────────────────

    def _ground_via_uniprot(self, entity_text: str) -> Optional[Dict[str, Any]]:
        """
        Ground a protein name via the UniProt REST search API.
        Returns a grounding dict or None on failure.
        """
        try:
            url = "https://rest.uniprot.org/uniprotkb/search"
            params = {
                "query": entity_text,
                "format": "json",
                "size": 1,
            }
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            if not results:
                return None
            hit = results[0]
            accession = hit.get("primaryAccession", "")
            recommended = (
                hit.get("proteinDescription", {})
                .get("recommendedName", {})
                .get("fullName", {})
                .get("value", entity_text)
            )
            return {
                "id": f"uniprot:{accession}",
                "canonical_name": recommended,
                "ontology": "UniProt",
                "confidence": CONFIDENCE_AUTHORITATIVE,
                "grounded": True,
                "source": "uniprot",
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "EntityNormalizer: UniProt lookup failed for {!r} — {}", entity_text, exc
            )
            return None

    # ── OLS grounding ─────────────────────────────────────────────────────────

    def _ground_via_ols(
        self, entity_text: str, ols_ontologies: Optional[str] = None, entity_type: str = "_default"
    ) -> Optional[Dict[str, Any]]:
        """
        Ground entity_text via EMBL-EBI OLS4 search API.
        ols_ontologies: comma-separated ontology IDs, e.g. "chebi,obi".
        Returns a grounding dict or None on failure.
        """
        try:
            params: Dict[str, Any] = {
                "q": entity_text,
                "rows": 1,
                "exact": "false",
            }
            if ols_ontologies:
                params["ontology"] = ols_ontologies

            # Retry logic for OLS API (handles transient 503/timeout)
            for attempt in range(3):
                try:
                    resp = requests.get(
                        "https://www.ebi.ac.uk/ols4/api/search",
                        params=params,
                        timeout=10,
                    )
                    resp.raise_for_status()
                    break
                except requests.exceptions.Timeout:
                    if attempt < 2:
                        time.sleep(2 ** attempt)
                        continue
                    raise
                except requests.exceptions.HTTPError as e:
                    if e.response.status_code in (503, 429) and attempt < 2:
                        time.sleep(2 ** attempt)
                        continue
                    raise

            data = resp.json()

            docs = (
                data.get("response", {}).get("docs", [])
            )
            if not docs:
                return None

            hit = docs[0]
            label = hit.get("label", entity_text)
            iri = hit.get("iri", "")
            ontology_name = hit.get("ontology_name", "OLS")

            # Validate returned ontology matches expected for this entity type
            EXPECTED_ONTOLOGIES = {
                "metabolite": ["chebi", "hmdb", "chembl"],
                "disease": ["mesh", "omim", "hp", "mondo", "doid"],
                "gene": ["ncbigene", "hgnc", "ensembl"],
                "protein": ["uniprot", "pr"],
                "body_site": ["uberon", "bto", "fma"],
                "immune_cell": ["cl"],
                "pathway": ["go", "pw", "reactome", "kegg"],
                "method": ["obi", "efo", "stato"],
                "drug": ["chebi", "chembl", "drugbank"],
                "treatment": ["mesh", "nci"],
            }
            hit_ontology = hit.get("ontology_name", "").lower()
            if ols_ontologies and entity_type in EXPECTED_ONTOLOGIES:
                expected = EXPECTED_ONTOLOGIES.get(entity_type, [])
                if expected and hit_ontology and not any(exp in hit_ontology for exp in expected):
                    logger.debug(
                        "EntityNormalizer: OLS returned ontology '{}' for entity_type '{}', expected one of {}. Rejecting.",
                        hit_ontology, entity_type, expected
                    )
                    return None

            return {
                "id": iri,
                "canonical_name": label,
                "ontology": ontology_name,
                "confidence": CONFIDENCE_AUTHORITATIVE,
                "grounded": True,
                "source": "ols",
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "EntityNormalizer: OLS lookup failed for {!r} — {}", entity_text, exc
            )
            return None

    # ── LLM grounding ─────────────────────────────────────────────────────────

    def _ground_via_llm(
        self, entity_text: str, entity_type: str
    ) -> Optional[Dict[str, Any]]:
        """
        Last-resort grounding via LLMGrounder (temperature=0, deterministic).
        Returns a grounding dict with confidence=CONFIDENCE_LLM and grounded=False
        (LLM output is not authoritative), or None if LLMGrounder is unavailable.
        """
        grounder = self._get_llm_grounder()
        if grounder is None:
            return None

        try:
            from semantic.candidate_store import CandidateEntity

            candidate = CandidateEntity(name=entity_text, entity_type=entity_type)
            result = grounder.resolve(candidate)

            ontology_id = result.get("ontology", "unknown")
            canonical = result.get("canonical", entity_text)

            # Never use LLM-suggested ontology IDs directly — they may be hallucinated
            # Only use the canonical name; the ID gets the llm: prefix to mark it non-authoritative
            return {
                "id": f"llm:{entity_text.lower().replace(' ', '_')}",
                "canonical_name": canonical,
                "ontology": None,       # LLM ontology IDs are unreliable — don't store them
                "ontology_hint": ontology_id if ontology_id != "unknown" else None,  # Store as hint only
                "confidence": CONFIDENCE_LLM,
                "grounded": False,
                "source": "llm",
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "EntityNormalizer: LLM grounding failed for {!r} — {}", entity_text, exc
            )
            return None

    # ── Ungrounded result helper ──────────────────────────────────────────────

    @staticmethod
    def _ungrounded(entity_text: str) -> Dict[str, Any]:
        return {
            "id": f"ungrounded:{entity_text.lower()}",
            "name": entity_text,
            "canonical_name": entity_text,
            "ontology": None,
            "confidence": CONFIDENCE_UNGROUNDED,
            "grounded": False,
            "source": "none",
        }

    # ── Main entry point ──────────────────────────────────────────────────────

    def normalize(self, entity_text: str, entity_type: str) -> Dict[str, Any]:
        """
        Normalize entity_text of entity_type to a canonical ontology grounding.

        Decision tree:
          1. Empty input → ungrounded dict
          2. Cache hit → return cached result
          3. YAML abbreviation map (taxon / disease only)
          4. Route to authoritative API via ONTOLOGY_ROUTING
          5. OLS cross-search fallback (for unknown types or API failure)
          6. LLM fallback (temperature=0, deterministic)
          7. All failed → ungrounded dict + failure log

        Returns dict with keys:
          id, name, canonical_name, ontology, confidence, grounded, source
        """
        # ── 1. Empty input ────────────────────────────────────────────────────
        if not entity_text or not entity_text.strip():
            return {
                "id": "ungrounded:empty",
                "name": entity_text or "",
                "canonical_name": "",
                "ontology": None,
                "confidence": CONFIDENCE_UNGROUNDED,
                "grounded": False,
                "source": "none",
            }

        entity_text = entity_text.strip()
        entity_type_lower = entity_type.lower() if entity_type else "_default"

        # ── 2. Cache lookup ───────────────────────────────────────────────────
        cached = self._cache_lookup(entity_text, entity_type_lower)
        if cached is not None:
            return cached

        # ── 3. YAML abbreviation map ──────────────────────────────────────────
        yaml_canonical = self._yaml_lookup(entity_text, entity_type_lower)
        if yaml_canonical:
            # Use the canonical form as the lookup term for the API
            lookup_text = yaml_canonical
        else:
            lookup_text = entity_text

        # ── 4. Route to authoritative API ────────────────────────────────────
        routing = ONTOLOGY_ROUTING.get(entity_type_lower, ONTOLOGY_ROUTING["_default"])
        api = routing.get("api", "ols_cross")
        result: Optional[Dict[str, Any]] = None

        if api in ("ncbi_taxonomy", "ncbi_mesh", "ncbi_gene"):
            ncbi_db = routing.get("ncbi_db", "taxonomy")
            result = self._ground_via_ncbi(lookup_text, ncbi_db)

        elif api == "uniprot":
            result = self._ground_via_uniprot(lookup_text)

        elif api == "ols":
            ols_ontologies = routing.get("ols_ontologies")
            result = self._ground_via_ols(lookup_text, ols_ontologies, entity_type=entity_type_lower)

        elif api == "ols_cross":
            # Unknown entity type — cross-search all OLS ontologies
            result = self._ground_via_ols(lookup_text, ols_ontologies=None, entity_type=entity_type_lower)

        # ── 5. OLS cross-search fallback (if primary API failed) ─────────────
        if result is None and api not in ("ols_cross",):
            logger.debug(
                "EntityNormalizer: primary API ({}) failed for {!r}; trying OLS cross-search",
                api,
                entity_text,
            )
            result = self._ground_via_ols(lookup_text, ols_ontologies=None, entity_type=entity_type_lower)

        # ── 6. LLM fallback ───────────────────────────────────────────────────
        if result is None:
            logger.debug(
                "EntityNormalizer: OLS cross-search failed for {!r}; trying LLM fallback",
                entity_text,
            )
            result = self._ground_via_llm(entity_text, entity_type_lower)

        # ── 7. All failed ─────────────────────────────────────────────────────
        if result is None:
            self._log_failure(
                entity_text=entity_text,
                entity_type=entity_type_lower,
                failure_reason=(
                    f"All grounding attempts failed: api={api}, "
                    "OLS cross-search, LLM fallback"
                ),
                attempted_matches=lookup_text,
            )
            result = self._ungrounded(entity_text)

        # Ensure "name" key is always present (some API helpers omit it)
        result.setdefault("name", entity_text)

        # Cache the result (including failures, to avoid re-querying)
        self._cache_store(entity_text, entity_type_lower, result)

        return result

    # ── Backward-compatible wrappers ──────────────────────────────────────────

    def normalize_taxon(self, x: str) -> Dict[str, Any]:
        """Backward-compatible wrapper — delegates to normalize(x, 'taxon')."""
        return self.normalize(x, "taxon")

    def normalize_disease(self, x: str) -> Dict[str, Any]:
        """Backward-compatible wrapper — delegates to normalize(x, 'disease')."""
        return self.normalize(x, "disease")
