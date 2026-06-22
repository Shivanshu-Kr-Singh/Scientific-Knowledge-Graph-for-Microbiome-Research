"""
semantic/llm_grounder.py — LLM-based biomedical entity grounder.

Uses Ollama as the sole LLM backend.
Uses _JsonFileCache for atomic, persistent caching of grounding results.

Requirements: 3.2, 3.6, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 6.1, 6.2, 6.3, 6.4,
              6.5, 6.6, 6.7, 11.2, 11.4, 14.2, 14.4
"""

import hashlib
import json
import logging
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from config import BACKEND_CONFIG
from semantic._cache import _JsonFileCache
from semantic.candidate_store import CandidateEntity
from semantic.ollama_client import OllamaClient, OllamaTimeoutError, OllamaUnavailableError

log = logging.getLogger(__name__)

# ─── Cache ────────────────────────────────────────────────────────────────────

_CACHE_PATH = Path(__file__).parent / "cache" / "llm_ground_cache.json"
_cache = _JsonFileCache(_CACHE_PATH)


# ─── Grounding prompt ─────────────────────────────────────────────────────────

def _build_prompt(entity: CandidateEntity) -> str:
    """
    Build the grounding prompt for the given entity.
    Returns the canonical biomedical name and the most appropriate ontology ID.
    Temperature=0 in OllamaClient ensures deterministic output.
    """
    ontology_hints = {
        "taxon":               "NCBI Taxonomy (format: NCBI:TXID e.g. NCBI:210 for H. pylori)",
        "disease":             "MeSH (format: MESH:DXXXXXX e.g. MESH:D015212 for IBD) or OMIM",
        "gene":                "NCBI Gene (format: NCBI_GENE:ID) or HGNC symbol",
        "protein":             "UniProt (format: UNIPROT:P12345) or canonical protein name",
        "metabolite":          "ChEBI (format: CHEBI:XXXXX e.g. CHEBI:17968 for butyrate) or HMDB",
        "pathway":             "KEGG (format: KEGG:map00XXX) or Reactome or GO",
        "body_site":           "UBERON (format: UBERON:XXXXXXX) or BTO",
        "immune_cell":         "Cell Ontology (format: CL:XXXXXXX)",
        "drug":                "ChEMBL (format: CHEMBL:XXXXX) or DrugBank",
        "treatment":           "MeSH (format: MESH:DXXXXXX) or NCI Thesaurus",
        "method":              "OBI (Ontology for Biomedical Investigations, format: OBI:XXXXXXX)",
        "clinical_outcome":    "MeSH or NCI Thesaurus",
        "biomarker":           "MeSH or relevant ontology; if purely computational (e.g. Shannon index), use 'statistical_measure'",
        "dietary_component":   "ChEBI or FoodOn (format: FOODON:XXXXX)",
        "environmental_factor":"ENVO or MeSH",
        "sequencing_platform": "OBI or EFO",
        "omics_feature":       "SO (Sequence Ontology, format: SO:XXXXXXX) or KEGG",
        "dataset":             "use accession prefix (e.g. SRA:SRP123456, GEO:GSE12345)",
        "population":          "MeSH or NCI Thesaurus population descriptor",
    }

    hint = ontology_hints.get(entity.entity_type.lower(), "most appropriate biomedical ontology")

    return (
        "/no_think\n"
        "You are a biomedical ontology grounding expert for human microbiome research.\n"
        "Your task: map a raw entity string to its official canonical name and ontology ID.\n\n"

        "RULES:\n"
        "1. canonical: the official scientific name (not abbreviation, not synonym)\n"
        "   - Prefer the full Latin binomial for taxa (e.g. 'Helicobacter pylori' not 'H. pylori')\n"
        "   - Prefer the MeSH descriptor name for diseases (e.g. 'Inflammatory Bowel Diseases' not 'IBD')\n"
        "   - Use the exact IUPAC name for metabolites when possible\n"
        "2. ontology: the best ontology ID in format PREFIX:ID\n"
        f"   - For {entity.entity_type}: use {hint}\n"
        "   - If no reliable ontology ID exists, use 'unknown'\n"
        "   - Do NOT guess or hallucinate IDs — return 'unknown' if uncertain\n\n"

        "EXAMPLES:\n"
        '  Entity: "h pylori", Type: taxon → {"canonical": "Helicobacter pylori", "ontology": "NCBI:210"}\n'
        '  Entity: "ibd", Type: disease → {"canonical": "Inflammatory Bowel Diseases", "ontology": "MESH:D015212"}\n'
        '  Entity: "butyrate", Type: metabolite → {"canonical": "butyrate", "ontology": "CHEBI:17968"}\n'
        '  Entity: "tlr4", Type: gene → {"canonical": "TLR4", "ontology": "NCBI_GENE:7099"}\n'
        '  Entity: "treg", Type: immune_cell → {"canonical": "regulatory T cell", "ontology": "CL:0000792"}\n'
        '  Entity: "gut-bone axis", Type: pathway → {"canonical": "gut-bone axis", "ontology": "unknown"}\n\n'

        "Return ONLY a JSON object, no markdown, no explanation:\n"
        '{"canonical": "official name here", "ontology": "PREFIX:ID or unknown"}\n\n'

        f"Entity: {entity.name}\n"
        f"Type: {entity.entity_type}"
    )


# ─── JSON parsing helpers ─────────────────────────────────────────────────────

def _strip_markdown_fence(raw: str) -> str:
    """
    Strip markdown code fences if present and log a WARNING.
    Returns the stripped string (may still be invalid JSON).
    """
    stripped = raw.strip()
    if stripped.startswith("```"):
        log.warning(
            "LLMGrounder: response begins with a markdown code fence — stripping fences"
        )
        lines = stripped.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines)
    return stripped


def _parse_response(raw: str, entity: CandidateEntity) -> dict | None:
    """
    Parse a raw LLM response string into a validated grounding dict.

    - Strips markdown fences if present.
    - Validates that both "canonical" and "ontology" keys are present and are strings.
    - Substitutes entity.name when canonical is an empty string.
    - Returns None and logs ERROR on any parse or validation failure.
    """
    cleaned = _strip_markdown_fence(raw)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        log.error("LLMGrounder: failed to parse JSON response. Raw response: %r", raw)
        return None

    if not isinstance(data, dict):
        log.error("LLMGrounder: parsed JSON is not a dict. Raw response: %r", raw)
        return None

    canonical = data.get("canonical")
    ontology = data.get("ontology")

    if not isinstance(canonical, str) or not isinstance(ontology, str):
        log.error(
            "LLMGrounder: response missing required keys or wrong types. Raw response: %r",
            raw,
        )
        return None

    if canonical == "":
        canonical = entity.name

    return {"canonical": canonical, "ontology": ontology}


# ─── LLMGrounder ──────────────────────────────────────────────────────────────


class LLMGrounder:
    """
    Resolves a CandidateEntity to its canonical form and ontology ID using Ollama.

    Results are cached in semantic/cache/llm_ground_cache.json.
    """

    def resolve(self, entity: CandidateEntity) -> dict:
        """
        Resolve *entity* to a dict with exactly the keys "canonical" and "ontology".

        Decision tree:
          1. Cache hit (valid entry) → return cached dict
          2. Call OllamaClient
             - On success: parse JSON, write cache, return result
             - On Ollama error: log ERROR, return fallback dict
          3. JSON parse failure: log ERROR, return fallback dict, do NOT cache
        """
        fallback = {"canonical": entity.name, "ontology": "unknown"}

        # ── 1. Cache lookup ───────────────────────────────────────────────────
        key = hashlib.md5(
            (entity.name + entity.entity_type).encode("utf-8")
        ).hexdigest()
        cache = _cache.load()

        if key in cache:
            cached_entry = cache[key]
            if (
                isinstance(cached_entry, dict)
                and isinstance(cached_entry.get("canonical"), str)
                and isinstance(cached_entry.get("ontology"), str)
            ):
                return {"canonical": cached_entry["canonical"], "ontology": cached_entry["ontology"]}

        # ── 2. Build prompt ───────────────────────────────────────────────────
        prompt = _build_prompt(entity)

        # ── 3. Call Ollama ────────────────────────────────────────────────────
        ollama_client = OllamaClient(BACKEND_CONFIG)
        model = BACKEND_CONFIG.ollama_grounding_model

        try:
            raw = ollama_client.generate(model, prompt)
        except (OllamaUnavailableError, OllamaTimeoutError) as exc:
            log.error(
                "LLMGrounder: %s: %s — returning fallback result",
                type(exc).__name__,
                exc,
            )
            return fallback

        # ── 4. Parse JSON response ────────────────────────────────────────────
        result = _parse_response(raw, entity)
        if result is None:
            return fallback

        # ── 5. Write to cache atomically ─────────────────────────────────────
        cache[key] = result
        _cache.save(cache)

        return result
