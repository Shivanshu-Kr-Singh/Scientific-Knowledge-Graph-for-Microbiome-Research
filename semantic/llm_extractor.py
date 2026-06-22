"""
semantic/llm_extractor.py — LLM-based biomedical entity and relation extractor.

Uses Ollama as the sole LLM backend.
Uses _JsonFileCache for atomic, persistent caching of extraction results.

Requirements: 3.1, 3.3, 3.4, 3.5, 4.1–4.8, 7.1–7.4, 8.1, 8.3, 11.1, 11.3, 11.5,
              14.1, 14.3, 14.5
"""

import hashlib
import json
import logging
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from config import BACKEND_CONFIG
from semantic._cache import _JsonFileCache
from semantic.candidate_store import CandidateEntity, CandidateRelation
from semantic.ollama_client import OllamaClient, OllamaTimeoutError, OllamaUnavailableError

log = logging.getLogger(__name__)

# ─── Cache ────────────────────────────────────────────────────────────────────

_CACHE_PATH = Path(__file__).parent / "cache" / "llm_extract_cache.json"
_cache = _JsonFileCache(_CACHE_PATH)


# ─── Extraction prompt ────────────────────────────────────────────────────────

def _build_prompt(text: str, known_entities: list = None) -> str:
    """
    Build a research-grade extraction prompt for human microbiome NER.
    Input text is truncated to 3,000 characters for speed.
    /no_think disables qwen3 chain-of-thought for faster responses.

    Includes: domain context, all 18 entity types with descriptions,
    confidence calibration, specificity rules, 1-shot example, and
    optional known-entities exclusion block.
    """
    truncated = text[:3000]

    # ── Known-entities exclusion block ────────────────────────────────────────
    if known_entities and len(known_entities) > 0:
        unique_known = list(dict.fromkeys(e.lower() for e in known_entities))[:50]
        known_block = (
            f"ALREADY EXTRACTED (do NOT repeat these):\n{unique_known}\n\n"
            "Your job: find entities NOT in the list above.\n\n"
        )
    else:
        known_block = ""

    # ── 1-shot example ────────────────────────────────────────────────────────
    few_shot = (
        'EXAMPLE INPUT: "Faecalibacterium prausnitzii was significantly depleted '
        "in IBD patients (p<0.01). Butyrate levels were reduced. 16S rRNA sequencing "
        "was performed on 50 fecal samples from Crohn's disease patients.\"\n"
        'EXAMPLE OUTPUT:\n'
        '{"entities": [\n'
        '  {"name": "Faecalibacterium prausnitzii", "type": "taxon", "confidence": 0.98},\n'
        '  {"name": "IBD", "type": "disease", "confidence": 0.95},\n'
        '  {"name": "butyrate", "type": "metabolite", "confidence": 0.92},\n'
        '  {"name": "16S rRNA sequencing", "type": "method", "confidence": 0.97},\n'
        '  {"name": "fecal", "type": "body_site", "confidence": 0.90},\n'
        "  {\"name\": \"Crohn's disease patients\", \"type\": \"population\", \"confidence\": 0.93}\n"
        '],\n'
        '"relations": [\n'
        '  {"subject": "Faecalibacterium prausnitzii", "predicate": "depleted in", "object": "IBD", "confidence": 0.95},\n'
        '  {"subject": "butyrate", "predicate": "decreased in", "object": "IBD patients", "confidence": 0.88}\n'
        '],\n'
        '"evidence": {"sample_size": 50, "study_design": "case-control", "population": "IBD patients"}}\n\n'
    )

    return (
        "/no_think\n"
        "You are a biomedical named entity recognition engine specialized in "
        "human microbiome research (2024-2026).\n\n"

        "PROJECT CONTEXT: Extract entities from microbiome research papers for a "
        "scientific knowledge graph. Focus on human microbiome studies covering "
        "gut, oral, skin, vaginal, and lung microbiomes and their associations "
        "with human health and disease.\n\n"

        "ENTITY TYPES — use ONLY these 18 types:\n"
        "  taxon             -> microbial taxa: species, genera, phyla, families "
        "(e.g. Lactobacillus rhamnosus, Firmicutes, Akkermansia muciniphila)\n"
        "  disease           -> medical conditions (e.g. IBD, Crohn's disease, T2D, obesity)\n"
        "  method            -> analytical/bioinformatics methods "
        "(e.g. 16S rRNA sequencing, QIIME2, DADA2, LEfSe)\n"
        "  body_site         -> anatomical locations (e.g. gut, colon, fecal, oral cavity)\n"
        "  treatment         -> interventions, drugs, probiotics, FMT, diet changes\n"
        "  dataset           -> named datasets or accession numbers (e.g. HMP, PRJNA123456)\n"
        "  metabolite        -> metabolic compounds (e.g. butyrate, TMAO, bile acids, SCFAs)\n"
        "  gene              -> genes, receptors, signaling molecules "
        "(e.g. TLR4, NOD2, NF-kB, IL-6, FXR)\n"
        "  protein           -> proteins, antibodies (e.g. zonulin, calprotectin, IgA)\n"
        "  biomarker         -> clinical/microbiome biomarkers "
        "(e.g. Shannon index, CRP, fecal calprotectin)\n"
        "  pathway           -> biological pathways (e.g. butyrate metabolism, TLR signaling)\n"
        "  population        -> study populations (e.g. IBD patients, healthy adults, neonates)\n"
        "  dietary_component -> food/dietary items (e.g. dietary fiber, inulin, polyphenols)\n"
        "  immune_cell       -> immune cells (e.g. Treg, Th17, macrophages, dendritic cells)\n"
        "  clinical_outcome  -> outcomes (e.g. remission, dysbiosis, mucosal healing)\n"
        "  environmental_factor -> exposures (e.g. antibiotic use, birth mode, breastfeeding)\n"
        "  sequencing_platform  -> platforms (e.g. Illumina MiSeq, PacBio, Oxford Nanopore)\n"
        "  omics_feature     -> omics units (e.g. OTU, ASV, MAG, KEGG ortholog)\n\n"

        "CONFIDENCE CALIBRATION:\n"
        "  0.95-1.00: exact named entity with full scientific name\n"
        "  0.85-0.94: named entity with clear context\n"
        "  0.70-0.84: entity mentioned but abbreviated or partially described\n"
        "  0.50-0.69: inferred or ambiguous entity\n"
        "  <0.50: do NOT include\n\n"

        "SPECIFICITY RULES:\n"
        "  - Prefer specific over general: 'Lactobacillus rhamnosus GG' > 'Lactobacillus'\n"
        "  - Include full strain names when present\n"
        "  - Use the exact name as it appears in the text\n"
        "  - Extract population descriptors (who was studied)\n"
        "  - Extract statistical evidence sentences for relations\n\n"

        f"{few_shot}"
        f"{known_block}"
        "INSTRUCTIONS:\n"
        "1. Extract ALL entities of the 18 types above from the TEXT\n"
        "2. Extract relations as (subject, predicate, object) triples\n"
        "3. Extract study evidence metadata\n"
        "4. Return ONLY valid JSON — no markdown, no explanation, no extra text\n\n"

        "REQUIRED OUTPUT FORMAT:\n"
        '{"entities": [{"name": "exact text span", "type": "one of 18 types", '
        '"confidence": 0.0-1.0}],\n'
        ' "relations": [{"subject": "entity name", "predicate": "verb phrase", '
        '"object": "entity name", "confidence": 0.0-1.0}],\n'
        ' "evidence": {"sample_size": integer_or_null, "study_design": "string", '
        '"population": "string"}}\n\n'
        "TEXT:\n"
        f"{truncated}"
    )


# ─── JSON parsing helpers ─────────────────────────────────────────────────────

def _strip_markdown_fence(raw: str) -> str:
    """
    Strip markdown code fences and qwen3 <think> blocks if present.
    Returns the stripped string (may still be invalid JSON).
    """
    import re as _re
    stripped = raw.strip()

    # Strip qwen3 <think>...</think> blocks
    stripped = _re.sub(r'<think>.*?</think>', '', stripped, flags=_re.DOTALL).strip()

    if stripped.startswith("```"):
        log.warning(
            "LLMExtractor: response begins with a markdown code fence — stripping fences"
        )
        lines = stripped.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines)
    return stripped


def _parse_response(raw: str) -> dict | None:
    """
    Parse a raw LLM response string into a dict.

    - Strips markdown fences if present (Req 3.3).
    - Handles qwen3 trailing newline padding.
    - Returns None and logs ERROR on JSON parse failure (Req 3.1).
    """
    cleaned = _strip_markdown_fence(raw)

    # qwen3 and some models append trailing \n\n\n... padding after the JSON.
    # Find the last closing brace and truncate there.
    last_brace = cleaned.rfind("}")
    if last_brace != -1:
        cleaned = cleaned[: last_brace + 1]

    # Also handle truncated JSON — find the first complete JSON object
    # by trying to parse progressively shorter strings if needed
    try:
        data = json.loads(cleaned)
        if not isinstance(data, dict):
            log.error("LLMExtractor: parsed JSON is not a dict. Raw response: %r", raw[:200])
            return None
        return data
    except json.JSONDecodeError:
        # Try to extract just the entities array if full parse fails
        import re as _re
        entities_match = _re.search(r'"entities"\s*:\s*(\[.*?\])', cleaned, _re.DOTALL)
        if entities_match:
            try:
                entities_list = json.loads(entities_match.group(1))
                return {"entities": entities_list, "relations": [], "evidence": {}}
            except json.JSONDecodeError:
                pass
        log.error("LLMExtractor: failed to parse JSON response. Raw response: %r", raw[:200])
        return None


def _build_result(data: dict) -> tuple[list[CandidateEntity], list[CandidateRelation]]:
    """
    Convert a parsed extraction schema dict into CandidateEntity / CandidateRelation lists.
    Applies safe defaults for missing or wrong-typed fields (Req 3.4, 3.5).
    """
    entities_raw = data.get("entities", [])
    if not isinstance(entities_raw, list):
        entities_raw = []

    relations_raw = data.get("relations", [])
    if not isinstance(relations_raw, list):
        relations_raw = []

    entities = [
        CandidateEntity(
            name=x.get("name", "") if isinstance(x.get("name"), str) else "",
            entity_type=x.get("type", "unknown") if isinstance(x.get("type"), str) else "unknown",
        )
        for x in entities_raw
        if isinstance(x, dict)
    ]

    relations = [
        CandidateRelation(
            subject=x.get("subject", "") if isinstance(x.get("subject"), str) else "",
            predicate=x.get("predicate", "") if isinstance(x.get("predicate"), str) else "",
            object=x.get("object", "") if isinstance(x.get("object"), str) else "",
            confidence=(
                x.get("confidence", 0.8)
                if isinstance(x.get("confidence"), (int, float))
                else 0.8
            ),
        )
        for x in relations_raw
        if isinstance(x, dict)
    ]

    return entities, relations




# ─── LLMExtractor ─────────────────────────────────────────────────────────────


class LLMExtractor:
    """
    Extracts biomedical entities and relations from text using Ollama.

    Results are cached in semantic/cache/llm_extract_cache.json.
    """

    def extract(
        self, text: str, known_entities: list = None
    ) -> tuple[list[CandidateEntity], list[CandidateRelation]]:
        """
        Extract entities and relations from *text*.

        known_entities: optional list of entity name strings already found by
                        regex/BioBERT. When provided, the LLM prompt instructs
                        the model to focus on novel entities not in this list.

        Decision tree:
          1. Empty / whitespace / None → return ([], [])
          2. Cache hit → return cached result
          3. Call OllamaClient
             - On success: parse JSON, write cache, return result
             - On Ollama error: log ERROR, return ([], [])
          4. JSON parse failure: log ERROR, return ([], []), do NOT write cache
        """
        # ── 1. Guard empty / whitespace / None input ──────────────────────────
        if not text or not text.strip():
            return [], []

        # ── 2. Cache lookup ───────────────────────────────────────────────────
        known_key = ",".join(sorted(known_entities)) if known_entities else ""
        key = hashlib.md5(f"{text}|{known_key}".encode("utf-8")).hexdigest()
        cache = _cache.load()
        if key in cache:
            cached_data = cache[key]
            if isinstance(cached_data, dict):
                return _build_result(cached_data)

        # ── 3. Build prompt ───────────────────────────────────────────────────
        prompt = _build_prompt(text, known_entities=known_entities)

        # ── 4. Call Ollama ────────────────────────────────────────────────────
        ollama_client = OllamaClient(BACKEND_CONFIG)
        model = BACKEND_CONFIG.ollama_extraction_model

        try:
            raw = ollama_client.generate(model, prompt)
        except (OllamaUnavailableError, OllamaTimeoutError) as exc:
            log.error(
                "LLMExtractor: %s: %s — returning empty result",
                type(exc).__name__,
                exc,
            )
            return [], []

        # ── 5. Parse JSON response ────────────────────────────────────────────
        data = _parse_response(raw)
        if data is None:
            return [], []

        # ── 6. Write to cache atomically ─────────────────────────────────────
        cache[key] = data
        _cache.save(cache)

        return _build_result(data)
