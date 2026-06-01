"""
OntologyTraverser — NCBI Taxonomy / MeSH hierarchy traversal for the
Deterministic Entity Resolution Pipeline.

When all direct matching strategies fail, this component queries the NCBI
Taxonomy hierarchy (for taxa) or the MeSH hierarchy (for diseases) to find
the nearest ancestor that exists in the CanonicalRegistry.  Traversal is
limited to 3 levels (parent, grandparent, great-grandparent).

Confidence by level:
    Level 1 (parent):            0.50
    Level 2 (grandparent):       0.40
    Level 3 (great-grandparent): 0.30

Graceful degradation: if the external ontology service is unavailable
(any network or HTTP error), a warning is logged and an empty list is
returned — the pipeline continues to the unresolved path.

Requirements: 13.1, 13.2, 13.3, 13.4, 13.5, 13.6
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING, List, Optional

import requests
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from entity_resolution.canonical_registry import CanonicalRegistry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# NCBI Entrez base URL for taxonomy lookups
_NCBI_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
_NCBI_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

# NLM MeSH REST API base URL
_MESH_LOOKUP_URL = "https://id.nlm.nih.gov/mesh/lookup/descriptor"
_MESH_SPARQL_URL = "https://id.nlm.nih.gov/mesh/sparql"

# HTTP request timeout in seconds
_HTTP_TIMEOUT = 10

# Maximum hierarchy levels to traverse (Requirement 13.2)
_MAX_LEVELS = 3


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


class OntologyCandidate(BaseModel):
    """
    A single candidate produced by the OntologyTraverser.

    Requirements: 13.3, 13.4
    """

    canonical_id: str
    hierarchy_level: int  # 1=parent, 2=grandparent, 3=great-grandparent
    grounding_confidence: float = Field(ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# OntologyTraverser
# ---------------------------------------------------------------------------


class OntologyTraverser:
    """
    NCBI Taxonomy / MeSH hierarchy traversal.

    Confidence by level:
    - Level 1 (parent):           0.50
    - Level 2 (grandparent):      0.40
    - Level 3 (great-grandparent): 0.30

    Preconditions for traverse():
    - surface_form is non-empty
    - entity_type is "taxon" or "disease"
    - registry is a CanonicalRegistry instance

    Postconditions for traverse():
    - Returns empty list if ontology service is unavailable (logs warning)
    - Returns empty list if no ancestor found within 3 levels
    - Returns at most one candidate (nearest ancestor in registry)
    - hierarchy_level is set to the level at which the match was found

    Requirements: 13.1, 13.2, 13.3, 13.4, 13.5, 13.6
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def traverse(
        self,
        surface_form: str,
        entity_type: str,
        registry: "CanonicalRegistry",
    ) -> List[OntologyCandidate]:
        """
        Find the nearest ancestor in the ontology hierarchy.

        Steps:
        1. Look up the surface_form in the external ontology to get its ID.
        2. Fetch the lineage / ancestor list.
        3. For each ancestor at level N (1–3), check if it exists in the
           CanonicalRegistry.
        4. Return the first match as an OntologyCandidate with
           hierarchy_level=N and grounding_confidence computed by
           compute_confidence(N).

        Graceful degradation: any network or HTTP exception is caught, a
        warning is logged, and [] is returned — the pipeline continues to
        the unresolved path (Requirement 13.1).

        Returns:
            A list containing at most one :class:`OntologyCandidate`.
            Returns [] when the service is unavailable or no ancestor is
            found within 3 levels (Requirements 13.1, 13.6).

        Requirements: 13.1, 13.2, 13.4, 13.5, 13.6
        """
        if not surface_form:
            return []

        entity_type_lower = entity_type.lower()

        try:
            if entity_type_lower == "taxon":
                return self._traverse_ncbi_taxonomy(surface_form, registry)
            elif entity_type_lower == "disease":
                return self._traverse_mesh(surface_form, registry)
            else:
                logger.warning(
                    "OntologyTraverser: unsupported entity_type '%s' for surface_form '%s'",
                    entity_type,
                    surface_form,
                )
                return []
        except requests.exceptions.RequestException as exc:
            # Network / HTTP error — graceful degradation (Requirement 13.1)
            logger.warning(
                "OntologyTraverser: ontology service unavailable for '%s' (%s): %s",
                surface_form,
                entity_type,
                exc,
            )
            return []
        except Exception as exc:
            # Any other unexpected error — log and degrade gracefully
            logger.warning(
                "OntologyTraverser: unexpected error for '%s' (%s): %s",
                surface_form,
                entity_type,
                exc,
            )
            return []

    @staticmethod
    def compute_confidence(hierarchy_level: int) -> float:
        """
        Compute grounding confidence for an ontology hierarchy match.

        Formula:
            confidence = 0.50 - (hierarchy_level - 1) * 0.10

        Valid for hierarchy_level in {1, 2, 3}:
        - Level 1 (parent):           0.50
        - Level 2 (grandparent):      0.40
        - Level 3 (great-grandparent): 0.30

        Args:
            hierarchy_level: Integer level (1, 2, or 3).

        Returns:
            Float confidence value.

        Requirements: 13.3
        """
        return 0.50 - (hierarchy_level - 1) * 0.10

    # ------------------------------------------------------------------
    # NCBI Taxonomy traversal
    # ------------------------------------------------------------------

    def _traverse_ncbi_taxonomy(
        self,
        surface_form: str,
        registry: "CanonicalRegistry",
    ) -> List[OntologyCandidate]:
        """
        Traverse the NCBI Taxonomy hierarchy for a taxon surface form.

        Steps:
        1. Use esearch to find the taxonomy ID for the surface_form.
        2. Use efetch to retrieve the full lineage XML.
        3. Parse the lineage to extract ancestor taxon IDs in order
           (nearest ancestor first).
        4. For each ancestor at level N (1–3), check the CanonicalRegistry.
        5. Return the first match.

        Raises:
            requests.exceptions.RequestException: on any HTTP/network error
            (caught by the caller traverse()).
        """
        # Step 1: search for the taxon ID
        taxon_id = self._ncbi_esearch(surface_form)
        if taxon_id is None:
            logger.debug(
                "OntologyTraverser: no NCBI Taxonomy ID found for '%s'", surface_form
            )
            return []

        # Step 2: fetch the lineage
        ancestors = self._ncbi_fetch_lineage(taxon_id)
        if not ancestors:
            logger.debug(
                "OntologyTraverser: no lineage found for taxon ID '%s' ('%s')",
                taxon_id,
                surface_form,
            )
            return []

        # Step 3–4: check each ancestor against the registry (up to 3 levels)
        return self._check_ancestors_in_registry(ancestors, registry)

    def _ncbi_esearch(self, surface_form: str) -> Optional[str]:
        """
        Search NCBI Taxonomy for the given surface form.

        Returns the first matching taxonomy ID as a string, or None if not found.

        Raises:
            requests.exceptions.RequestException: on network/HTTP error.
        """
        params = {
            "db": "taxonomy",
            "term": surface_form,
            "retmode": "json",
            "retmax": "1",
        }
        response = requests.get(
            _NCBI_ESEARCH_URL, params=params, timeout=_HTTP_TIMEOUT
        )
        response.raise_for_status()

        data = response.json()
        id_list = data.get("esearchresult", {}).get("idlist", [])
        if not id_list:
            return None
        return id_list[0]

    def _ncbi_fetch_lineage(self, taxon_id: str) -> List[str]:
        """
        Fetch the lineage for a given NCBI Taxonomy ID.

        Returns a list of ancestor taxon IDs ordered from nearest (parent)
        to most distant, limited to _MAX_LEVELS entries.

        Raises:
            requests.exceptions.RequestException: on network/HTTP error.
        """
        params = {
            "db": "taxonomy",
            "id": taxon_id,
            "rettype": "xml",
            "retmode": "xml",
        }
        response = requests.get(
            _NCBI_EFETCH_URL, params=params, timeout=_HTTP_TIMEOUT
        )
        response.raise_for_status()

        # Parse the XML response to extract the lineage
        # The NCBI Taxonomy XML contains a <LineageEx> element with <Taxon> children
        # ordered from root to the immediate parent of the queried taxon.
        try:
            root = ET.fromstring(response.text)
        except ET.ParseError as exc:
            logger.warning(
                "OntologyTraverser: failed to parse NCBI Taxonomy XML for ID '%s': %s",
                taxon_id,
                exc,
            )
            return []

        # Extract lineage taxon IDs from <LineageEx><Taxon><TaxId>...</TaxId></Taxon>...
        # The lineage is ordered root → ... → parent, so we reverse to get
        # nearest ancestor first.
        lineage_ids: List[str] = []
        lineage_ex = root.find(".//LineageEx")
        if lineage_ex is not None:
            for taxon_elem in lineage_ex.findall("Taxon"):
                tax_id_elem = taxon_elem.find("TaxId")
                if tax_id_elem is not None and tax_id_elem.text:
                    lineage_ids.append(tax_id_elem.text.strip())

        # Reverse so nearest ancestor (parent) is first
        lineage_ids.reverse()

        # Limit to _MAX_LEVELS
        return lineage_ids[:_MAX_LEVELS]

    # ------------------------------------------------------------------
    # MeSH traversal
    # ------------------------------------------------------------------

    def _traverse_mesh(
        self,
        surface_form: str,
        registry: "CanonicalRegistry",
    ) -> List[OntologyCandidate]:
        """
        Traverse the MeSH hierarchy for a disease surface form.

        Steps:
        1. Use the NLM MeSH lookup API to find the descriptor ID for the
           surface_form.
        2. Use the MeSH SPARQL endpoint to retrieve the parent descriptors
           up to 3 levels.
        3. For each ancestor at level N (1–3), check the CanonicalRegistry.
        4. Return the first match.

        Raises:
            requests.exceptions.RequestException: on any HTTP/network error
            (caught by the caller traverse()).
        """
        # Step 1: look up the MeSH descriptor ID
        descriptor_id = self._mesh_lookup_descriptor(surface_form)
        if descriptor_id is None:
            logger.debug(
                "OntologyTraverser: no MeSH descriptor found for '%s'", surface_form
            )
            return []

        # Step 2: fetch ancestors via SPARQL
        ancestors = self._mesh_fetch_ancestors(descriptor_id)
        if not ancestors:
            logger.debug(
                "OntologyTraverser: no MeSH ancestors found for descriptor '%s' ('%s')",
                descriptor_id,
                surface_form,
            )
            return []

        # Step 3–4: check each ancestor against the registry (up to 3 levels)
        return self._check_ancestors_in_registry(ancestors, registry)

    def _mesh_lookup_descriptor(self, surface_form: str) -> Optional[str]:
        """
        Look up a MeSH descriptor ID for the given surface form.

        Uses the NLM MeSH lookup API:
        https://id.nlm.nih.gov/mesh/lookup/descriptor?label=<term>&match=exact

        Returns the descriptor ID (e.g. "D006262") or None if not found.

        Raises:
            requests.exceptions.RequestException: on network/HTTP error.
        """
        params = {
            "label": surface_form,
            "match": "exact",
            "limit": "1",
        }
        response = requests.get(
            _MESH_LOOKUP_URL, params=params, timeout=_HTTP_TIMEOUT
        )
        response.raise_for_status()

        data = response.json()
        if not data:
            return None

        # The API returns a list of objects with a "resource" field like
        # "http://id.nlm.nih.gov/mesh/D006262"
        first = data[0] if isinstance(data, list) else None
        if first is None:
            return None

        resource = first.get("resource", "")
        # Extract the descriptor ID from the URI
        if "/" in resource:
            descriptor_id = resource.rsplit("/", 1)[-1]
            return descriptor_id if descriptor_id else None

        return None

    def _mesh_fetch_ancestors(self, descriptor_id: str) -> List[str]:
        """
        Fetch the parent descriptors for a MeSH descriptor via SPARQL.

        Returns a list of ancestor descriptor IDs ordered from nearest
        (parent) to most distant, limited to _MAX_LEVELS entries.

        Raises:
            requests.exceptions.RequestException: on network/HTTP error.
        """
        # SPARQL query to get up to 3 levels of parents
        # We use a property path to traverse broaderDescriptor relationships
        sparql_query = f"""
        PREFIX mesh: <http://id.nlm.nih.gov/mesh/>
        PREFIX meshv: <http://id.nlm.nih.gov/mesh/vocab#>

        SELECT ?level1 ?level2 ?level3 WHERE {{
            OPTIONAL {{
                mesh:{descriptor_id} meshv:broaderDescriptor ?level1 .
                OPTIONAL {{
                    ?level1 meshv:broaderDescriptor ?level2 .
                    OPTIONAL {{
                        ?level2 meshv:broaderDescriptor ?level3 .
                    }}
                }}
            }}
        }}
        LIMIT 1
        """

        headers = {"Accept": "application/sparql-results+json"}
        response = requests.get(
            _MESH_SPARQL_URL,
            params={"query": sparql_query, "format": "JSON"},
            headers=headers,
            timeout=_HTTP_TIMEOUT,
        )
        response.raise_for_status()

        data = response.json()
        bindings = data.get("results", {}).get("bindings", [])
        if not bindings:
            return []

        ancestors: List[str] = []
        binding = bindings[0]

        for level_key in ("level1", "level2", "level3"):
            if level_key in binding:
                uri = binding[level_key].get("value", "")
                if "/" in uri:
                    ancestor_id = uri.rsplit("/", 1)[-1]
                    if ancestor_id:
                        ancestors.append(ancestor_id)

        return ancestors[:_MAX_LEVELS]

    # ------------------------------------------------------------------
    # Shared helper: check ancestors against registry
    # ------------------------------------------------------------------

    def _check_ancestors_in_registry(
        self,
        ancestors: List[str],
        registry: "CanonicalRegistry",
    ) -> List[OntologyCandidate]:
        """
        Check each ancestor ID against the CanonicalRegistry.

        Iterates through the ancestors list (nearest first) and for each
        ancestor at level N (1-indexed), checks whether the ancestor's ID
        exists in the registry.  Returns the first match as an
        OntologyCandidate with hierarchy_level=N.

        If no ancestor is found within _MAX_LEVELS, returns [] (Req 13.6).

        Args:
            ancestors: List of ancestor IDs ordered nearest-first, already
                       limited to at most _MAX_LEVELS entries.
            registry:  The CanonicalRegistry to check against.

        Returns:
            A list containing at most one OntologyCandidate.

        Requirements: 13.2, 13.3, 13.4, 13.6
        """
        for level, ancestor_id in enumerate(ancestors, start=1):
            if level > _MAX_LEVELS:
                break  # Requirement 13.2: traverse at most 3 levels

            # Check if this ancestor exists in the registry by its canonical_id
            record = registry.lookup_by_canonical_id(ancestor_id)
            if record is not None:
                confidence = self.compute_confidence(level)
                return [
                    OntologyCandidate(
                        canonical_id=ancestor_id,
                        hierarchy_level=level,
                        grounding_confidence=confidence,
                    )
                ]

        # No ancestor found within 3 levels (Requirement 13.6)
        return []
