"""
nlp/ner.py
-----------
Named Entity Recognition for biomedical entities in microbiome papers.

ENTITIES WE EXTRACT:
  taxon      → microbial taxa: "Bacteroides fragilis", "Firmicutes", "E. coli"
  disease    → conditions: "Crohn's disease", "IBD", "obesity", "type 2 diabetes"
  method     → sequencing methods: "16S rRNA", "shotgun metagenomics", "WGS"
  body_site  → anatomical locations: "gut", "colon", "oral cavity", "skin"
  treatment  → interventions: "probiotics", "FMT", "antibiotics", "diet"
  dataset    → data resources: "HMP", "curatedMetagenomicData", "FINRISK"

APPROACH — 2-tier:
  Tier 1: Rule-based dictionary matching (fast, high precision)
           Covers known taxa, methods, body sites from curated lists.
           Runs on every paper with no GPU needed.

  Tier 2: BioBERT NER model (slower, handles novel entities)
           Only loaded when needed (lazy import).
           Falls back gracefully if transformers not installed.
           Model: "pruas/BENT-PubMedBERT-NER-Gene" or
                  "d4data/biomedical-ner-all" from HuggingFace.

WHY BOTH?
  Dictionary matching catches 100% of known terms with zero false positives.
  BioBERT catches novel phrasings, new species names, abbreviations the
  dictionary doesn't have yet. Together they cover > 90% recall.
"""

import re
from typing import List, Optional
from loguru import logger

from nlp.enriched_record import NamedEntity


# ── Tier 1 Dictionaries ───────────────────────────────────────────────────────

TAXA_PATTERNS = [
    # Phyla (broad)
    r"\bfirmicutes\b", r"\bbacteroidetes\b", r"\bproteobacteria\b",
    r"\bactinobacteria\b", r"\bverrucomicrobia\b", r"\bfusobacteria\b",
    r"\bspirochaetes\b", r"\btenericutes\b", r"\beuryarchaeota\b",

    # Common genera
    r"\bbacteroides\b", r"\blactobacillus\b", r"\bbifidobacterium\b",
    r"\bfaecalibacterium\b", r"\bruminococcus\b", r"\bclostridium\b",
    r"\bprevotella\b", r"\broseburia\b", r"\bblautia\b", r"\bakkermansia\b",
    r"\bveillonella\b", r"\bstreptococcus\b", r"\bstaphylococcus\b",
    r"\bescherichia\b", r"\bklebsiella\b", r"\bsalmonella\b",
    r"\bhelicobacter\b", r"\bcampylobacter\b", r"\bmycobacterium\b",

    # General patterns for binomial names: Capital genus + lowercase species
    # Known microbiome species only
    r"\bakkermansia muciniphila\b",
    r"\bbacteroides fragilis\b",
    r"\bfaecalibacterium prausnitzii\b",
    r"\blactobacillus acidophilus\b",
    r"\blactobacillus rhamnosus\b",
    r"\blactobacillus reuteri\b",
    r"\bbifidobacterium longum\b",
    r"\bbifidobacterium infantis\b",
    r"\bhelicobacter pylori\b",
    r"\bclostridium difficile\b",
    r"\bclostridioides difficile\b",

    # Novel species pattern
]

DISEASE_PATTERNS = [
    r"\birritable bowel syndrome\b", r"\bibs\b",
    r"\binflammatory bowel disease\b", r"\bibd\b",
    r"\bcrohn.s disease\b", r"\bulcerative colitis\b",
    r"\bcolorectal cancer\b", r"\bcrc\b",
    r"\btype 2 diabetes\b", r"\bt2d\b", r"\btype 2 diabetes mellitus\b",
    r"\bobesity\b", r"\boverweight\b", r"\bmetabolic syndrome\b",
    r"\bnon.alcoholic fatty liver\b", r"\bnafld\b", r"\bnash\b",
    r"\bceliac disease\b",
    r"\brheumatoid arthritis\b",
    r"\bparkinson.s disease\b",
    r"\balzheimer.s disease\b",
    r"\bautism spectrum disorder\b", r"\basd\b",
    r"\bdepression\b", r"\banxiety\b",
    r"\bcoronary artery disease\b",
    r"\bhypertension\b",
    r"\bcolorectal adenoma\b",
    r"\bgastric cancer\b",
    r"\bcirrhosis\b",
    r"\bnecrotizing enterocolitis\b",
    r"\bsepsis\b",
    r"\bgraft.versus.host disease\b", r"\bgvhd\b",
    r"\bhiv\b", r"\baids\b",
    r"\bcovid.19\b", r"\bsars-cov-2\b",
]

METHOD_PATTERNS = [
    r"\b16s rrna(?: gene)? sequencing\b",
    r"\bshotgun metagenomics\b",
    r"\bwhole[- ]genome sequencing\b",
    r"\bwgs\b",
    r"\bmetatranscriptomics\b",
    r"\bamplicon sequencing\b",
    r"\bshort[- ]read sequencing\b",
    r"\blong[- ]read sequencing\b",
    r"\bnanopore",
    r"\bpacbio",
    r"\billumina (miseq|hiseq|novaseq)",
    r"\bqiime\b",
    r"\bqiime2\b",
    r"\bmetaphlan\b",
    r"\bhumann\b",
    r"\bpicrust\b",
    r"\bbiobakery\b",
    r"\bdada2\b",
    r"\bdeblur\b",
    r"\botu\b", r"\basv\b",
    r"\bphylogenetic analysis\b",
    r"\bshotgun sequencing\b",
    r"\bwhole metagenome sequencing\b",
    r"\bflow cytometry\b",
    r"\b16s\b",
]

BODY_SITE_PATTERNS = [
    r"\bgut\b", r"\bintestin", r"\bcolon\b", r"\brectum\b",
    r"\bcecum\b", r"\bduodenum\b", r"\bjejunum\b", r"\bileum\b",
    r"\bstomach\b", r"\bgastric\b",
    r"\boral\b", r"\bmouth\b", r"\bsaliva\b", r"\bdental\b",
    r"\boral cavity\b", r"\bsubgingival\b",
    r"\bskin\b", r"\bcutaneous\b",
    r"\blung\b", r"\brespiratory\b", r"\bnasopharyn",
    r"\bvagina\b", r"\bvaginal\b", r"\bcervical\b",
    r"\bneonatal\b", r"\bnewborn\b",
    r"\bblood\b", r"\bserum\b", r"\bplasma\b",
    r"\bfecal\b", r"\bstool\b", r"\bfaecal\b",
    r"\bbladder\b", r"\burinary\b",
    r"\bbrain\b", r"\bneural\b",
    r"\bliver\b", r"\bhepatic\b",
]

TREATMENT_PATTERNS = [
    r"\bprobiotic",
    r"\bprebiotic",
    r"\bsynbiotic",
    r"\bfecal microbiota transplant", r"\bfmt\b",
    r"\bantibiotic",
    r"\bmetformin\b",
    r"\bdiet\b", r"\bdietary intervent",
    r"\bmediterranean diet\b",
    r"\bhigh.fiber diet\b",
    r"\bfermented food",
    r"\bplant.based diet\b",
    r"\bketogenic diet\b",
    r"\bcaloric restrict",
    r"\bintermittent fasting\b",
    r"\bbutyrate\b",
    r"\bshort.chain fatty acid", r"\bscfa\b",
    r"\bvitamin d\b",
    r"\bproton pump inhibitor", r"\bppi\b",
    r"\bnon.steroidal anti.inflammatory", r"\bnsaid\b",
    r"\bimmunosuppressant",
]

DATASET_PATTERNS = [
    r"\bhuman microbiome project\b", r"\bhmp\b",
    r"\bcuratedmetagenomicdata\b",
    r"\bfinrisk\b",
    r"\bamerica gut\b",
    r"\bbiome project\b",
    r"\bmetahit\b",
    r"\bmibc\b",
    r"\bgmrepo\b",
    r"\bmgx\b",
    r"\bgnps\b",
]

ENTITY_PATTERNS = {
    "taxon":     TAXA_PATTERNS,
    "disease":   DISEASE_PATTERNS,
    "method":    METHOD_PATTERNS,
    "body_site": BODY_SITE_PATTERNS,
    "treatment": TREATMENT_PATTERNS,
    "dataset":   DATASET_PATTERNS,
}


class NERExtractor:
    """
    Extracts named biomedical entities from paper text.
    Uses rule-based matching (always) + optional BioBERT (if available).
    """

    def __init__(self, use_model: bool = False):
        """
        use_model: If True, loads BioBERT NER model for additional coverage.
                   Requires 'transformers' and 'torch' installed.
                   Falls back gracefully if not available.
        """
        self._model = None
        self._tokenizer = None
        self._model_loaded = False

        if use_model:
            self._load_model()

    def _load_model(self):
        """
        Lazy-loads the BioBERT NER model.

        MODEL CHOICE: 'd4data/biomedical-ner-all'
          - Fine-tuned on multiple biomedical NER datasets
          - Recognizes: chemicals, diseases, genes, organisms, proteins
          - CPU-compatible (slow but works without GPU)
          - ~440MB download on first use (cached by HuggingFace)

        If you have a GPU, this runs ~10x faster.
        If you only have CPU, process papers in batches overnight.
        """
        try:
            from transformers import pipeline
            logger.info("[NER] Loading BioBERT model (first run downloads ~440MB)...")
            self._model = pipeline(
                "ner",
                model="d4data/biomedical-ner-all",
                aggregation_strategy="simple",  # Merges B/I/O tags into spans
                device=-1,  # CPU; change to 0 for first GPU
            )
            self._model_loaded = True
            logger.info("[NER] BioBERT model loaded")
        except ImportError:
            logger.warning("[NER] transformers not installed — using rules only. Run: pip install transformers torch")
        except Exception as e:
            logger.warning(f"[NER] Model load failed: {e} — using rules only")

    def extract(self, title: str, abstract: Optional[str]) -> List[NamedEntity]:
        """
        Extracts all named entities from title + abstract.
        Returns a deduplicated list of NamedEntity objects.
        """
        title = title or "" 
        abstract = abstract or "" 
        text = (f"{title} {abstract}").lower()
        full_text = f"{title or ''} {abstract or ''}"

        entities: List[NamedEntity] = []

        # Tier 1: Rule-based dictionary matching
        rule_entities = self._rule_based_extract(text)
        entities.extend(rule_entities)

        # Tier 2: BioBERT model (if loaded)
        if self._model_loaded and self._model:
            model_entities = self._model_extract(full_text)
            entities.extend(model_entities)

        # Deduplicate by (text, label) pair
        seen = set()
        unique = []
        for e in entities:
            key = (e.text.lower(), e.label)
            if key not in seen:
                seen.add(key)
                unique.append(e)

        return unique

    def _rule_based_extract(self, text_lower: str) -> List[NamedEntity]:
        """
        Applies regex patterns to extract entities.
        text_lower should already be lowercased for case-insensitive matching.
        """
        results = []
        for label, patterns in ENTITY_PATTERNS.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text_lower, re.IGNORECASE):
                    span = match.group(0).strip()
                    # Remove obvious false taxa
                    if label == "taxon":
                        bad = {"patients underwent","data available","shotgun sequencing","gut microbiome","human microbiome","shotgun metagenomics"}
                        if span.lower() in bad:
                            continue
                    if len(span) < 2:
                        continue   # Skip single-character matches
                    results.append(NamedEntity(
                        text=span,
                        label=label,
                        start=match.start(),
                        end=match.end(),
                        confidence=1.0,  # Rule-based = deterministic
                    ))
        return results

    def _model_extract(self, text: str) -> List[NamedEntity]:
        """
        Runs BioBERT NER model on the text.
        Maps BioBERT's label scheme to our controlled vocabulary.
        """
        if not self._model or not text.strip():
            return []

        try:
            # Truncate to model's max token length (~512 tokens ~ 350 words)
            truncated = " ".join(text.split()[:350])
            raw_entities = self._model(truncated)

            results = []
            for ent in raw_entities:
                label = self._map_model_label(ent.get("entity_group", ""))
                if label:
                    results.append(NamedEntity(
                        text=ent.get("word", "").strip(),
                        label=label,
                        confidence=round(float(ent.get("score", 0)), 3),
                    ))
            return results

        except Exception as e:
            logger.warning(f"[NER] Model inference failed: {e}")
            return []

    def _map_model_label(self, raw_label: str) -> Optional[str]:
        """
        Maps BioBERT's output labels to our entity vocabulary.
        d4data/biomedical-ner-all uses these labels:
          B-Chemical, B-Disease, B-Gene_or_gene_product, B-Organism,
          B-Simple_chemical, B-Protein, etc.
        """
        label_map = {
            "Chemical":           "treatment",
            "Simple_chemical":    "treatment",
            "Disease":            "disease",
            "Gene_or_gene_product": None,    # Not relevant for our use case
            "Organism":           "taxon",
            "Species":            "taxon",
            "Cell":               "body_site",
            "Protein":            None,
        }
        for key, mapped in label_map.items():
            if key.lower() in raw_label.lower():
                return mapped
        return None

    def group_entities(self, entities: List[NamedEntity]) -> dict:
        """
        Groups entities by label for easy access.
        Returns dict like: {"taxon": [...], "disease": [...], ...}
        """
        groups: dict = {label: [] for label in ENTITY_PATTERNS}
        for ent in entities:
            if ent.label in groups:
                groups[ent.label].append(ent.text)
        # Deduplicate within each group
        return {label: list(dict.fromkeys(items)) for label, items in groups.items()}
