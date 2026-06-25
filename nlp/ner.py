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

APPROACH — 3-tier:
Tier 1: Rule-based dictionary matching (fast, high precision)
Tier 2: BioBERT NER model (slower, handles novel entities)
Tier 3: LLM extraction via Ollama (highest recall, catches novel relationships)
"""

import re
from typing import List, Optional
from loguru import logger
import threading

from nlp.enriched_record import NamedEntity

# ── GPU resource management ───────────────────────────────────────────────────
# When BioBERT and Ollama both run on the same GPU, they must take turns.
# This semaphore ensures only ONE GPU operation happens at a time:
#   - BioBERT inference acquires it before calling self._model()
#   - Ollama NER acquires it before calling the LLM
# Additionally, _OLLAMA_NER_LOCK ensures only 1 thread calls Ollama at a time
# (Ollama is single-threaded internally — concurrent calls just queue anyway
# but cause timeouts because each waits 120s for its turn).
_GPU_SEMAPHORE   = threading.Semaphore(1)   # 1 GPU op at a time
_OLLAMA_NER_LOCK = threading.Lock()          # 1 Ollama NER call at a time


# ── Tier 1 Dictionaries ───────────────────────────────────────────────────────

TAXA_PATTERNS = [
    # ── Phyla ──
    r"\bfirmicutes\b", r"\bbacteroidetes\b", r"\bproteobacteria\b",
    r"\bactinobacteria\b", r"\bactinobacteriota\b", r"\bverrucomicrobia\b",
    r"\bfusobacteria\b", r"\bfusobacteriota\b", r"\bspirochaetes\b",
    r"\btenericutes\b", r"\beuryarchaeota\b", r"\bthaumarchaeota\b",
    r"\bchloroflexi\b", r"\bplanctomycetes\b", r"\bdeferribacteres\b",
    r"\bsynergistetes\b", r"\bthermodesulfobacteria\b", r"\bcyanobacteria\b",
    r"\bdeltaproteobacteria\b", r"\bgammaproteobacteria\b",
    r"\bepsilonproteobacteria\b", r"\balphaproteobacteria\b",
    r"\bbetaproteobacteria\b", r"\bacidobacteria\b", r"\bgemmatimonadetes\b",
    # ── Classes / Orders ──
    r"\bclostridia\b", r"\bbacilli\b", r"\berysipelotrichia\b",
    r"\bnegativicutes\b", r"\bbacteroidia\b", r"\bflavobacteriia\b",
    r"\bactinomycetia\b", r"\bcoriobacteriia\b",
    # ── Families ──
    r"\blachnospiraceae\b", r"\bruminococcaceae\b", r"\bbacteroidaceae\b",
    r"\bprevotellaceae\b", r"\brikenellaceae\b", r"\bporphyromonadaceae\b",
    r"\bbifidobacteriaceae\b", r"\blactobacillaceae\b", r"\bstreptococcaceae\b",
    r"\benterobacteriaceae\b", r"\bveillonellaceae\b", r"\beggerthellaceae\b",
    r"\bchristensenellaceae\b", r"\boscillospiraceae\b", r"\bfirmicutes\b",
    r"\berysipelotrichaceae\b", r"\bpeptostreptococcaceae\b",
    r"\bclostridiales\b", r"\bbacteroidales\b", r"\blactobacillales\b",
    # ── Genera ──
    r"\bbacteroides\b", r"\blactobacillus\b", r"\bbifidobacterium\b",
    r"\bfaecalibacterium\b", r"\bruminococcus\b", r"\bclostridium\b",
    r"\bclostridioides\b", r"\bprevotella\b", r"\broseburia\b", r"\bblautia\b",
    r"\bakkermansia\b", r"\bveillonella\b", r"\bstreptococcus\b",
    r"\bstaphylococcus\b", r"\bescherichia\b", r"\bklebsiella\b",
    r"\bsalmonella\b", r"\bhelicobacter\b", r"\bcampylobacter\b",
    r"\bmycobacterium\b", r"\benterococcus\b", r"\benterobacter\b",
    r"\bproteus\b", r"\bpseudomonas\b", r"\bacinetobacter\b",
    r"\bfusobacterium\b", r"\bporphyromonas\b", r"\btreponema\b",
    r"\btannerella\b", r"\balistipes\b", r"\bparabacteroides\b",
    r"\bbarnesiella\b", r"\bodorribacter\b", r"\bphascolarctobacterium\b",
    r"\bdialister\b", r"\bmegamonas\b", r"\bcoprococcus\b",
    r"\bdorea\b", r"\beubacterium\b", r"\banaerostipes\b",
    r"\bbutyrivibrio\b", r"\bsubdoligranulum\b", r"\boscillibacter\b",
    r"\bflavonifractor\b", r"\banaerotruncus\b", r"\bholdemanella\b",
    r"\bholdemania\b", r"\bpeptostreptococcus\b", r"\bfinegoldia\b",
    r"\banaerococcus\b", r"\bpeptoniphilus\b", r"\bgemella\b",
    r"\bgranulicatella\b", r"\babiotrophia\b", r"\bsutterella\b",
    r"\bbilophila\b", r"\bdesulfovibrio\b", r"\bdesulfobacter\b",
    r"\bchristensenella\b", r"\bsporobacter\b", r"\bintestinimonas\b",
    r"\bbutyricoccus\b", r"\bpseudoflavonifractor\b", r"\bpseudobutyrivibrio\b",
    r"\bturicibacter\b", r"\berysipelotrichaceae\b", r"\bcatenibacterium\b",
    r"\bholdemanella\b", r"\bcloacibacillus\b", r"\bsynergistes\b",
    r"\bpyloribacter\b", r"\bselenomonas\b", r"\bmegasphaera\b",
    r"\bmitsuokella\b", r"\bwolinella\b", r"\bsuccinivibrio\b",
    r"\bsucciniclasticum\b", r"\banaerobiospirillum\b",
    r"\bsaccharibacteria\b", r"\btm7\b", r"\bcpr\b",
    r"\bspiroplasma\b", r"\bmycoplasma\b", r"\bureaplasma\b",
    r"\bchlamydia\b", r"\bchlamydophila\b", r"\brickettsiales\b",
    r"\bwolbachia\b", r"\bborrelia\b", r"\bleptospira\b",
    r"\blisteria\b", r"\bbacillus\b", r"\bpaenibacillus\b",
    r"\blactococcus\b", r"\bleuconostoc\b", r"\bpediococcus\b",
    r"\bweissella\b", r"\boenococcus\b", r"\btetragenococcus\b",
    r"\baeriscardovia\b", r"\bscardovia\b", r"\balloscardovia\b",
    r"\bparascardovia\b", r"\bbombiscardovia\b",
    r"\bpropionibacterium\b", r"\bcutibacterium\b", r"\bcorynebacterium\b",
    r"\bbrevibacterium\b", r"\bmicrococcus\b", r"\bkocuria\b",
    r"\brothibacterium\b", r"\bcellulomonas\b", r"\bnocardia\b",
    r"\bstreptomyces\b", r"\bsaccharopolyspora\b",
    # ── Species ──
    r"\bakkermansia muciniphila\b",
    r"\bbacteroides fragilis\b", r"\bbacteroides thetaiotaomicron\b",
    r"\bbacteroides uniformis\b", r"\bbacteroides vulgatus\b",
    r"\bbacteroides ovatus\b", r"\bbacteroides dorei\b",
    r"\bfaecalibacterium prausnitzii\b",
    r"\blactobacillus acidophilus\b", r"\blactobacillus rhamnosus\b",
    r"\blactobacillus reuteri\b", r"\blactobacillus plantarum\b",
    r"\blactobacillus casei\b", r"\blactobacillus fermentum\b",
    r"\blactobacillus gasseri\b", r"\blactobacillus johnsonii\b",
    r"\blactobacillus salivarius\b", r"\blactobacillus crispatus\b",
    r"\blactobacillus iners\b", r"\blactobacillus jensenii\b",
    r"\blactobacillus helveticus\b", r"\blactobacillus delbrueckii\b",
    r"\blactobacillus bulgaricus\b", r"\blactobacillus brevis\b",
    r"\blactobacillus paracasei\b", r"\blactobacillus pentosus\b",
    r"\bbifidobacterium longum\b", r"\bbifidobacterium infantis\b",
    r"\bbifidobacterium breve\b", r"\bbifidobacterium adolescentis\b",
    r"\bbifidobacterium bifidum\b", r"\bbifidobacterium animalis\b",
    r"\bbifidobacterium lactis\b", r"\bbifidobacterium pseudolongum\b",
    r"\bhelicobacter pylori\b",
    r"\bclostridium difficile\b", r"\bclostridioides difficile\b",
    r"\bclostridium perfringens\b", r"\bclostridium butyricum\b",
    r"\bclostridium sporogenes\b",
    r"\bruminococcus gnavus\b", r"\bruminococcus torques\b",
    r"\bruminococcus bromii\b", r"\bruminococcus champanellensis\b",
    r"\broseburia intestinalis\b", r"\broseburia hominis\b",
    r"\broseburia inulinivorans\b",
    r"\bblautia obeum\b", r"\bblautia producta\b", r"\bblautia wexlerae\b",
    r"\bprevotella copri\b", r"\bprevotella stercorea\b",
    r"\bprevotella melaninogenica\b",
    r"\balistipes putredinis\b", r"\balistipes shahii\b",
    r"\bparabacteroides distasonis\b", r"\bparabacteroides merdae\b",
    r"\beubacterium rectale\b", r"\beubacterium hallii\b",
    r"\beubacterium eligens\b", r"\beubacterium ventriosum\b",
    r"\bcoprococcus eutactus\b", r"\bcoprococcus comes\b",
    r"\bdorea longicatena\b", r"\bdorea formicigenerans\b",
    r"\banaerostipes caccae\b", r"\banaerostipes hadrus\b",
    r"\bbutyrivibrio fibrisolvens\b",
    r"\bsubdoligranulum variabile\b",
    r"\boscillibacter valericigenes\b",
    r"\bdesulfovibrio piger\b",
    r"\bbilophila wadsworthia\b",
    r"\bchristensenella minuta\b",
    r"\benterococcus faecalis\b", r"\benterococcus faecium\b",
    r"\bstreptococcus thermophilus\b", r"\bstreptococcus salivarius\b",
    r"\bstreptococcus mutans\b", r"\bstreptococcus pyogenes\b",
    r"\bescherichia coli\b", r"\bklebsiella pneumoniae\b",
    r"\bklebsiella oxytoca\b", r"\benterobacter cloacae\b",
    r"\bproteus mirabilis\b", r"\bpseudomonas aeruginosa\b",
    r"\bacinetobacter baumannii\b",
    r"\bfusobacterium nucleatum\b", r"\bfusobacterium periodonticum\b",
    r"\bporphyromonas gingivalis\b", r"\btannerella forsythia\b",
    r"\btreponema denticola\b",
    r"\bsutterella wadsworthensis\b",
    r"\blisteria monocytogenes\b",
    r"\bsalmonella enterica\b", r"\bsalmonella typhi\b",
    r"\bcampylobacter jejuni\b", r"\bcampylobacter coli\b",
    r"\bmycobacterium tuberculosis\b", r"\bmycobacterium avium\b",
    r"\bstaphylococcus aureus\b", r"\bstaphylococcus epidermidis\b",
    # ── Archaea ──
    r"\bmethanobrevibacter smithii\b", r"\bmethanobrevibacter\b",
    r"\bmethanosphaera stadtmanae\b", r"\bmethanosphaera\b",
    r"\bmethanomassiliicoccus\b", r"\bmethanomassiliicoccales\b",
    # ── Fungi / Mycobiome ──
    r"\bcandida albicans\b", r"\bcandida tropicalis\b", r"\bcandida glabrata\b",
    r"\bcandida\b", r"\bsaccharomyces cerevisiae\b", r"\bsaccharomyces\b",
    r"\baspergillus\b", r"\bmalassezia\b", r"\bcryptococcus\b",
    r"\bpneumocystis\b", r"\bmycobiome\b", r"\bfungal microbiome\b",
    # ── Viruses / Virome ──
    r"\bvirome\b", r"\bbacteriophage\b", r"\bphage\b",
    r"\bcrass-like phage\b", r"\bcrassphage\b",
    r"\benterovirus\b", r"\bnorovirus\b", r"\brotavirus\b",
    r"\badenovirus\b", r"\bcoronavirus\b",
    # ── General microbiome terms ──
    r"\bgut microbiome\b", r"\bgut microbiota\b", r"\bintestinal microbiome\b",
    r"\bintestinal microbiota\b", r"\bfecal microbiota\b",
    r"\boral microbiome\b", r"\boral microbiota\b",
    r"\bskin microbiome\b", r"\blung microbiome\b",
    r"\bvaginal microbiome\b", r"\bvaginal microbiota\b",
    r"\bnasopharyngeal microbiome\b", r"\burinary microbiome\b",
    r"\bmicrobial community\b", r"\bmicrobial diversity\b",
    r"\bmicrobial composition\b", r"\bmicrobial abundance\b",
    r"\balpha diversity\b", r"\bbeta diversity\b",
    r"\bshannon diversity\b", r"\bsimpson diversity\b",
    r"\bchao1\b", r"\bfaith.s pd\b", r"\bunifrac\b",
    r"\bbray.curtis\b", r"\bjaccard\b",
    r"\bdysbiosis\b", r"\bmicrobial dysbiosis\b",
    r"\bmicrobial imbalance\b", r"\bmicrobial shift\b",
    r"\btaxonomic profil", r"\bphylogenetic diversit",
    r"\bcore microbiome\b", r"\bcommensal\b", r"\bsymbiont\b",
    r"\bpathobiont\b", r"\bopportunistic pathogen\b",

    # ── Reclassified genera (IJSEM/NCBI updates 2020–2024) ───────────────────
    # Lactobacillaceae reclassification (Zheng et al. 2020) — old names still
    # appear in many papers; both old and new names must be caught.
    r"\bligilactobacillus\b",          # was Lactobacillus (e.g. L. animalis)
    r"\blacticaseibacillus\b",         # was Lactobacillus casei group
    r"\blactiplantibacillus\b",        # was L. plantarum group
    r"\blevilactobacillus\b",          # was L. brevis group
    r"\blimosilactobacillus\b",        # was L. reuteri group (L. reuteri, L. fermentum)
    r"\bpediococcus acidilactici\b",
    r"\bpediococcus pentosaceus\b",
    r"\blactiplantibacillus plantarum\b",
    r"\blacticaseibacillus rhamnosus\b",
    r"\blacticaseibacillus paracasei\b",
    r"\blacticaseibacillus casei\b",
    r"\bligilactobacillus salivarius\b",
    r"\blimosilactobacillus reuteri\b",
    r"\blimosilactobacillus fermentum\b",
    r"\blevilactobacillus brevis\b",

    # Lachnospiraceae reclassifications
    r"\bblautia wexlerae\b",           # confirmed human gut species
    r"\bblautia massiliensis\b",
    r"\bagathobacter rectalis\b",       # was Eubacterium rectale
    r"\bagathobacter hallii\b",         # was Eubacterium hallii
    r"\bagathobacter\b",
    r"\bfusicatenibacter saccharivorans\b",
    r"\bfusicatenibacter\b",
    r"\bpseudobutyrivibrio ruminis\b",
    r"\bpseudobutyrivibrio\b",
    r"\bmonoglobus pectinilyticus\b",
    r"\bmonoglobus\b",
    r"\bacetatifactor muris\b",
    r"\bacetatifactor\b",

    # Ruminococcaceae reclassifications
    r"\bcaproiciproducens galactitolivorans\b",
    r"\bcaproiciproducens\b",
    r"\bpseudoflavonifractor capillosus\b",
    r"\bpseudoflavonifractor\b",

    # Erysipelotrichaceae reclassifications
    r"\bthomassilella stercoricola\b",
    r"\bthomassilella\b",
    r"\bcatenibacterium mitsuokai\b",
    r"\bcatenibacterium\b",
    r"\bclostridium innocuum\b",        # reclassified to Erysipelatoclostridium
    r"\berysipelatoclostridium innocuum\b",
    r"\berysipelatoclostridium\b",

    # Clostridiaceae / Peptostreptococcaceae reclassifications
    r"\btyzzerella nexilis\b",
    r"\btyzzerella\b",
    r"\bflintibacter butyricus\b",
    r"\bflintibacter\b",
    r"\bbombella intestinalis\b",
    r"\bbombella\b",
    r"\banaerofustis stercorihominis\b",
    r"\banaerofustis\b",

    # Bacteroidaceae reclassifications
    r"\bparabacteroides goldsteinii\b",
    r"\balistipes finegoldii\b",
    r"\balistipes onderdonkii\b",
    r"\balistipes timonensis\b",

    # Verrucomicrobiota (new phylum name)
    r"\bverrucomicrobiota\b",           # previously Verrucomicrobia
    r"\bakkermansiaceae\b",

    # Phylum-level renames (GTDB-based nomenclature appearing in recent papers)
    r"\bbackwardsbacteria\b",
    r"\bpseudomonadota\b",              # new NCBI name for Proteobacteria
    r"\bbacillota\b",                   # new NCBI name for Firmicutes
    r"\bbacteroidota\b",                # new NCBI name for Bacteroidetes
    r"\bactinomycetota\b",              # new NCBI name for Actinobacteria
    r"\bfusobacteriota\b",              # new NCBI name for Fusobacteria
    r"\bspirochaetota\b",               # new NCBI name for Spirochaetes
    r"\bcampylobacterota\b",            # new NCBI name for Epsilonproteobacteria
    r"\bdesulfobacterota\b",            # new NCBI name for Deltaproteobacteria
    r"\bmyxococcota\b",
    r"\bsynergistota\b",                # new NCBI name for Synergistetes
    r"\bthermotogota\b",
]

DISEASE_PATTERNS = [
    # ── GI diseases ──
    r"\birritable bowel syndrome\b", r"\bibs\b",
    r"\binflammatory bowel disease\b", r"\bibd\b",
    r"\bcrohn.s disease\b", r"\bcrohn disease\b",
    r"\bulcerative colitis\b", r"\buc\b",
    r"\bcolorectal cancer\b", r"\bcrc\b", r"\bcolon cancer\b",
    r"\brectal cancer\b", r"\bcolorectal adenoma\b",
    r"\bgastric cancer\b", r"\bstomach cancer\b",
    r"\bgastroesophageal reflux\b", r"\bgerd\b",
    r"\bpeptic ulcer\b", r"\bgastric ulcer\b", r"\bduodenal ulcer\b",
    r"\bconstipation\b", r"\bdiarrhea\b", r"\bdiarrhoea\b",
    r"\bceliac disease\b", r"\bcoeliac disease\b",
    r"\bsmall intestinal bacterial overgrowth\b", r"\bsibo\b",
    r"\bintestinal permeability\b", r"\bleaky gut\b",
    r"\bgastrointestinal infection\b", r"\bgastroenteritis\b",
    r"\bclostridium difficile infection\b", r"\bcdi\b",
    r"\bpouchitis\b", r"\bproctitis\b", r"\benteritis\b",
    r"\bcolitis\b", r"\benterocolitis\b",
    r"\bnecrotizing enterocolitis\b", r"\bnec\b",
    r"\bshort bowel syndrome\b",
    # ── Metabolic diseases ──
    r"\btype 2 diabetes\b", r"\bt2d\b", r"\btype 2 diabetes mellitus\b",
    r"\btype 1 diabetes\b", r"\bt1d\b", r"\btype 1 diabetes mellitus\b",
    r"\bgestational diabetes\b",
    r"\bobesity\b", r"\boverweight\b", r"\bbmi\b",
    r"\bmetabolic syndrome\b", r"\bmetabolic disorder\b",
    r"\bnon.alcoholic fatty liver\b", r"\bnafld\b", r"\bnash\b",
    r"\bnon.alcoholic steatohepatitis\b",
    r"\balcoholic liver disease\b", r"\bald\b",
    r"\bcirrhosis\b", r"\bhepatic cirrhosis\b", r"\bliver cirrhosis\b",
    r"\bhepatic encephalopathy\b",
    r"\bhyperuricemia\b", r"\bgout\b",
    r"\bhyperlipidemia\b", r"\bdyslipidemia\b",
    r"\bhypertension\b", r"\bhigh blood pressure\b",
    r"\binsulin resistance\b", r"\bglucose intolerance\b",
    r"\bprediabetes\b",
    # ── Autoimmune / Inflammatory ──
    r"\brheumatoid arthritis\b", r"\bra\b",
    r"\bsystemic lupus erythematosus\b", r"\bsle\b", r"\blupus\b",
    r"\bmultiple sclerosis\b", r"\bms\b",
    r"\bpsoriasis\b", r"\bpsoriatic arthritis\b",
    r"\bankylosing spondylitis\b",
    r"\bsjogren.s syndrome\b",
    r"\bscleroderma\b", r"\bsystemic sclerosis\b",
    r"\bvasculitis\b", r"\bsarcoidosis\b",
    r"\bceliac disease\b",
    r"\btype 1 diabetes\b",
    r"\bthyroiditis\b", r"\bhashimoto\b", r"\bgraves disease\b",
    r"\batopic dermatitis\b", r"\beczema\b",
    r"\basthma\b", r"\ballergy\b", r"\ballergic rhinitis\b",
    r"\bfood allergy\b", r"\bfood intolerance\b",
    # ── Neurological / Psychiatric ──
    r"\bparkinson.s disease\b", r"\bparkinson disease\b",
    r"\balzheimer.s disease\b", r"\balzheimer disease\b",
    r"\bdementia\b", r"\bcognitive decline\b",
    r"\bautism spectrum disorder\b", r"\basd\b", r"\bautism\b",
    r"\bdepression\b", r"\bmajor depressive disorder\b", r"\bmdd\b",
    r"\banxiety\b", r"\banxiety disorder\b",
    r"\bschizophrenia\b", r"\bbipolar disorder\b",
    r"\badhd\b", r"\battention deficit\b",
    r"\bgut.brain axis\b", r"\bgut brain axis\b",
    r"\bneuroinflammation\b",
    r"\bstroke\b", r"\bcerebrovascular\b",
    r"\bepilepsy\b", r"\bseizure\b",
    r"\bamyotrophic lateral sclerosis\b", r"\bals\b",
    # ── Cardiovascular ──
    r"\bcoronary artery disease\b", r"\bcad\b",
    r"\bheart failure\b", r"\bcardiac failure\b",
    r"\batrial fibrillation\b",
    r"\bmyocardial infarction\b", r"\bheart attack\b",
    r"\batherosclerosis\b", r"\barteriosclerosis\b",
    r"\bthrombosis\b", r"\bvenous thromboembolism\b",
    r"\bperipheral artery disease\b",
    # ── Cancer ──
    r"\bcolorectal cancer\b", r"\bgastric cancer\b",
    r"\bpancreatic cancer\b", r"\bhepatocellular carcinoma\b",
    r"\bhcc\b", r"\bliver cancer\b",
    r"\blung cancer\b", r"\bbreast cancer\b",
    r"\bprostate cancer\b", r"\bcervical cancer\b",
    r"\boral cancer\b", r"\besophageal cancer\b",
    r"\bbladder cancer\b", r"\bkidney cancer\b",
    r"\bleukemia\b", r"\blymphoma\b", r"\bmyeloma\b",
    r"\bmelanoma\b", r"\bcolorectal neoplasm\b",
    # ── Infectious diseases ──
    r"\bsepsis\b", r"\bbacteremia\b", r"\bsepticemia\b",
    r"\bgraft.versus.host disease\b", r"\bgvhd\b",
    r"\bhiv\b", r"\baids\b", r"\bhiv infection\b",
    r"\bcovid.19\b", r"\bsars.cov.2\b", r"\bcoronavirus disease\b",
    r"\binfluenza\b", r"\bpneumonia\b",
    r"\btuberculosis\b", r"\btb\b",
    r"\bmalaria\b", r"\bviral hepatitis\b",
    r"\bhepatitis b\b", r"\bhepatitis c\b",
    r"\bclostridium difficile\b", r"\bcdi\b",
    r"\bvaginosis\b", r"\bbacterial vaginosis\b", r"\bbv\b",
    r"\bcandidiasis\b", r"\bvulvovaginal candidiasis\b",
    # ── Kidney / Urological ──
    r"\bchronic kidney disease\b", r"\bckd\b",
    r"\bend.stage renal disease\b", r"\besrd\b",
    r"\burinary tract infection\b", r"\buti\b",
    r"\bnephrolithiasis\b", r"\bkidney stone\b",
    # ── Respiratory ──
    r"\bchronic obstructive pulmonary disease\b", r"\bcopd\b",
    r"\bcystic fibrosis\b", r"\bcf\b",
    r"\bbronchiectasis\b", r"\bpneumonia\b",
    # ── Pediatric / Neonatal ──
    r"\bnecrotizing enterocolitis\b", r"\bnec\b",
    r"\bcolic\b", r"\binfant colic\b",
    r"\bchildhood obesity\b", r"\bpediatric ibd\b",
    r"\bneonatal sepsis\b",
    # ── Other ──
    r"\bfatigue\b", r"\bchronic fatigue syndrome\b", r"\bcfs\b",
    r"\bfibromyalgia\b",
    r"\bosteoporosis\b", r"\bosteopenia\b",
    r"\bsarcopenia\b", r"\bfrailty\b",
    r"\bpolycystic ovary syndrome\b", r"\bpcos\b",
    r"\bendometriosis\b",
    r"\bpreeclampsia\b", r"\bgestational hypertension\b",
    r"\bpreterm birth\b", r"\bpreterm\b",
]

METHOD_PATTERNS = [
    # ── Sequencing methods ──
    r"\b16s rrna(?: gene)? sequencing\b", r"\b16s sequencing\b", r"\b16s\b",
    r"\bshotgun metagenomics\b", r"\bshotgun sequencing\b",
    r"\bwhole[- ]genome sequencing\b", r"\bwgs\b",
    r"\bwhole metagenome sequencing\b", r"\bwms\b",
    r"\bmetatranscriptomics\b", r"\bmetatranscriptomic\b",
    r"\bmetaproteomics\b", r"\bmetaproteomic\b",
    r"\bmetabolomics\b", r"\bmetabolomic\b",
    r"\bmetagenomics\b", r"\bmetagenomic\b",
    r"\bamplicon sequencing\b", r"\bv3.v4 region\b", r"\bv4 region\b",
    r"\bv1.v3 region\b", r"\bv3 region\b",
    r"\bshort[- ]read sequencing\b", r"\blong[- ]read sequencing\b",
    r"\bnanopore sequencing\b", r"\bnanopore\b",
    r"\bpacbio\b", r"\bpacific biosciences\b",
    r"\billumina miseq\b", r"\billumina hiseq\b", r"\billumina novaseq\b",
    r"\billumina nextseq\b", r"\billumina\b",
    r"\bion torrent\b", r"\bion pgm\b",
    r"\bsanger sequencing\b",
    r"\bsingle.cell sequencing\b", r"\bsc.seq\b",
    r"\bspatial transcriptomics\b",
    r"\brna.seq\b", r"\brnaseq\b", r"\btranscriptomics\b",
    r"\bchip.seq\b", r"\batac.seq\b",
    # ── Bioinformatics pipelines ──
    r"\bqiime\b", r"\bqiime2\b", r"\bqiime 2\b",
    r"\bmetaphlan\b", r"\bmetaphlan2\b", r"\bmetaphlan3\b", r"\bmetaphlan4\b",
    r"\bhumann\b", r"\bhumann2\b", r"\bhumann3\b",
    r"\bpicrust\b", r"\bpicrust2\b",
    r"\bbiobakery\b",
    r"\bdada2\b", r"\bdeblur\b",
    r"\botu\b", r"\basv\b", r"\bamplicon sequence variant\b",
    r"\botu clustering\b", r"\botu picking\b",
    r"\bphyloseq\b", r"\bvegan\b", r"\bade4\b",
    r"\bkraken\b", r"\bkraken2\b", r"\bbraken\b",
    r"\bkaiju\b", r"\bcentrifuge\b", r"\bdiamond\b",
    r"\bmegan\b", r"\bmothur\b",
    r"\bsortmerna\b", r"\bvsearch\b", r"\bswarm\b",
    r"\bpear\b", r"\btrimmomatic\b", r"\bfastp\b", r"\btrim galore\b",
    r"\bbowtie\b", r"\bbwa\b", r"\bsamtools\b",
    r"\bspades\b", r"\bmegahit\b", r"\bmetabat\b", r"\bconcoct\b",
    r"\bprodigal\b", r"\bprokka\b", r"\banvio\b",
    r"\bcheckm\b", r"\bgtdb.tk\b", r"\bgtdbtk\b",
    r"\bfastqc\b", r"\bmultiqc\b",
    r"\bpiphillin\b", r"\btax4fun\b",
    r"\bpermanova\b", r"\banosim\b", r"\bmantel test\b",
    r"\brandom forest\b", r"\bsvm\b", r"\bsupport vector machine\b",
    r"\blinear discriminant analysis\b", r"\blda\b", r"\blefse\b",
    r"\bdeseq2\b", r"\bedger\b", r"\blimma\b",
    r"\bmaaslin\b", r"\bmaaslin2\b",
    r"\bsparcc\b", r"\bspiec.easi\b", r"\bconet\b",
    # ── Analytical methods ──
    r"\bphylogenetic analysis\b", r"\bphylogenetics\b",
    r"\bflow cytometry\b", r"\bfacs\b",
    r"\belisa\b", r"\bwestern blot\b", r"\bpcr\b", r"\bqpcr\b",
    r"\brt.pcr\b", r"\bddpcr\b",
    r"\bnmr\b", r"\bnuclear magnetic resonance\b",
    r"\bmass spectrometry\b", r"\blc.ms\b", r"\bgc.ms\b",
    r"\bscfa analysis\b", r"\bshort.chain fatty acid analysis\b",
    r"\bbile acid analysis\b", r"\btryptophan metabolite\b",
    r"\buntargeted metabolomics\b", r"\btargeted metabolomics\b",
    r"\bculture.based\b", r"\bculture independent\b",
    r"\bgerm.free\b", r"\bgnotobiotic\b",
    r"\bhumanized mouse\b", r"\bcolonization\b",
    r"\bfecal transplant\b", r"\bfmt\b",
    r"\bex vivo\b", r"\bin vitro\b", r"\bin vivo\b",
    r"\brandomized controlled trial\b", r"\brct\b",
    r"\bcohort study\b", r"\bcase.control\b", r"\bcross.sectional\b",
    r"\blongitudinal study\b", r"\bprospective\b", r"\bretrospective\b",
    r"\bmeta.analysis\b", r"\bsystematic review\b",
    r"\bmendelian randomization\b", r"\bgenome.wide association\b", r"\bgwas\b",
]

BODY_SITE_PATTERNS = [
    # ── GI tract ──
    r"\bgut\b", r"\bintestin", r"\bcolon\b", r"\brectum\b",
    r"\bcecum\b", r"\bcaecum\b", r"\bduodenum\b", r"\bjejunum\b", r"\bileum\b",
    r"\bstomach\b", r"\bgastric\b", r"\bgastrointestinal\b", r"\bgi tract\b",
    r"\bsmall intestine\b", r"\blarge intestine\b",
    r"\bappendix\b", r"\bileocecal\b", r"\bileostomy\b", r"\bcolostomy\b",
    r"\bcolon mucosa\b", r"\bintestinal mucosa\b", r"\bcolonic mucosa\b",
    r"\bgut epithelium\b", r"\bintestinal epithelium\b",
    r"\bgut lumen\b", r"\bintestinal lumen\b",
    r"\bmucus layer\b", r"\bmucus barrier\b",
    r"\bgut wall\b", r"\bgut barrier\b", r"\bintestinal barrier\b",
    r"\btight junction\b",
    # ── Oral ──
    r"\boral\b", r"\bmouth\b", r"\bsaliva\b", r"\bdental\b",
    r"\boral cavity\b", r"\bsubgingival\b", r"\bsupragingival\b",
    r"\bgingival\b", r"\bperiodontal\b", r"\bdentition\b",
    r"\btongue\b", r"\bbuccal\b", r"\bpharynx\b", r"\btonsil\b",
    # ── Skin ──
    r"\bskin\b", r"\bcutaneous\b", r"\bdermis\b", r"\bepidermis\b",
    r"\bsebaceous\b", r"\bsweat gland\b", r"\bfollicle\b",
    r"\bscalp\b", r"\baxilla\b", r"\bgroin\b",
    # ── Respiratory ──
    r"\blung\b", r"\brespiratory\b", r"\bnasopharyn",
    r"\bairway\b", r"\bbronch", r"\balveol",
    r"\bnasal\b", r"\bnose\b", r"\bsinuses\b",
    r"\btrachea\b", r"\blarynx\b",
    # ── Urogenital ──
    r"\bvagina\b", r"\bvaginal\b", r"\bcervical\b", r"\bcervix\b",
    r"\buterus\b", r"\bendometrium\b", r"\bplacenta\b",
    r"\bbladder\b", r"\burinary\b", r"\burethra\b",
    r"\bkidney\b", r"\brenal\b",
    r"\bprostate\b", r"\bsemen\b", r"\btestis\b",
    # ── Neonatal / Pediatric ──
    r"\bneonatal\b", r"\bnewborn\b", r"\binfant\b",
    r"\bbreast milk\b", r"\bhuman milk\b", r"\bcolostrum\b",
    r"\bumbilical\b", r"\bamniot",
    # ── Blood / Systemic ──
    r"\bblood\b", r"\bserum\b", r"\bplasma\b",
    r"\bperipheral blood\b", r"\bwhole blood\b",
    r"\blymph node\b", r"\bspleen\b", r"\bthymus\b",
    r"\bbone marrow\b",
    # ── Stool / Fecal ──
    r"\bfecal\b", r"\bstool\b", r"\bfaecal\b", r"\bfeces\b", r"\bfaeces\b",
    r"\bfecal sample\b", r"\bstool sample\b",
    # ── Liver / Biliary ──
    r"\bliver\b", r"\bhepatic\b", r"\bbile\b", r"\bbiliary\b",
    r"\bgallbladder\b", r"\bbile duct\b",
    # ── Brain / CNS ──
    r"\bbrain\b", r"\bneural\b", r"\bcns\b", r"\bcerebral\b",
    r"\bcerebrospinal fluid\b", r"\bcsf\b",
    r"\bgut.brain\b", r"\bgut brain\b",
    # ── Adipose / Muscle ──
    r"\badipose\b", r"\bfat tissue\b", r"\bvisceral fat\b",
    r"\bsubcutaneous fat\b", r"\bmuscle\b", r"\bskeletal muscle\b",
    # ── Pancreas ──
    r"\bpancreas\b", r"\bpancreatic\b", r"\bislets\b",
]

TREATMENT_PATTERNS = [
    # ── Probiotics / Prebiotics ──
    r"\bprobiotic", r"\bprebiotic", r"\bsynbiotic", r"\bpostbiotic",
    r"\bparaprobiotic\b", r"\bpsychobiotic\b",
    r"\blive biotherapeutic\b", r"\bbiotherapeutic\b",
    r"\bfermented milk\b", r"\byogurt\b", r"\bkefir\b",
    r"\bkombucha\b", r"\bsauerkraut\b", r"\bkimchi\b",
    r"\bfermented food\b", r"\bfermented beverage\b",
    # ── FMT ──
    r"\bfecal microbiota transplant", r"\bfmt\b",
    r"\bfecal transplant\b", r"\bstool transplant\b",
    r"\bmicrobiota transplant\b",
    r"\bfecal bacteriotherapy\b",
    # ── Antibiotics ──
    r"\bantibiotic", r"\bantimicrobial",
    r"\bamoxicillin\b", r"\bmetronidazole\b", r"\bciprofloxacin\b",
    r"\bvancomycin\b", r"\brifaximin\b", r"\bneomycin\b",
    r"\bampicillin\b", r"\btetracycline\b", r"\bdoxycycline\b",
    r"\bclindamycin\b", r"\bazithromycin\b", r"\bclarithromycin\b",
    r"\bfluoroquinolone\b", r"\bcephalosporin\b", r"\bpenicillin\b",
    r"\bcarbapenems\b", r"\bcolistin\b", r"\bpolymyxin\b",
    r"\blinezolid\b", r"\bdaptomycin\b", r"\bfidaxomicin\b",
    # ── Diet interventions ──
    r"\bdiet\b", r"\bdietary intervent",
    r"\bmediterranean diet\b", r"\bhigh.fiber diet\b",
    r"\bplant.based diet\b", r"\bketogenic diet\b",
    r"\blow.carbohydrate diet\b", r"\blow.fat diet\b",
    r"\bvegan diet\b", r"\bvegetarian diet\b",
    r"\bgluten.free diet\b", r"\bdairy.free diet\b",
    r"\bwestern diet\b", r"\bhigh.fat diet\b",
    r"\bcaloric restrict", r"\bintermittent fasting\b",
    r"\btime.restricted eating\b", r"\btime.restricted feeding\b",
    r"\bfasting\b", r"\bfood supplementation\b",
    r"\bdietary fiber\b", r"\bfiber supplement\b",
    r"\binulin\b", r"\bfructooligosaccharide\b", r"\bfos\b",
    r"\bgalactooligosaccharide\b", r"\bgos\b",
    r"\bpectin\b", r"\bpsyllium\b", r"\bbetaglucan\b",
    r"\bresistant starch\b", r"\bpolyphenol\b",
    r"\bquercetin\b", r"\bresveratrol\b", r"\bcurcumin\b",
    # ── Metabolites / Supplements ──
    r"\bbutyrate\b", r"\bpropionate\b", r"\bacetate\b",
    r"\bshort.chain fatty acid", r"\bscfa\b",
    r"\bbile acid\b", r"\bsecondary bile acid\b",
    r"\btryptophan\b", r"\bindole\b", r"\bserotonin\b",
    r"\bgaba\b", r"\bdopamine\b",
    r"\bvitamin d\b", r"\bvitamin b12\b", r"\bvitamin k\b",
    r"\bfolate\b", r"\bfolic acid\b", r"\bzinc\b", r"\biron\b",
    r"\bmagnesium\b", r"\bcalcium\b", r"\bselenium\b",
    r"\bomega.3\b", r"\bfish oil\b", r"\bdha\b", r"\bepa\b",
    # ── Drugs ──
    r"\bmetformin\b", r"\binsulin\b", r"\bglp.1\b",
    r"\bsemaglutide\b", r"\bliraglutide\b", r"\bexenatide\b",
    r"\bproton pump inhibitor\b", r"\bppi\b",
    r"\bomeprazole\b", r"\bpantoprazole\b", r"\besomeprazole\b",
    r"\bnon.steroidal anti.inflammatory\b", r"\bnsaid\b",
    r"\bibuprofen\b", r"\baspirin\b", r"\bnaproxen\b",
    r"\bimmunosuppressant\b", r"\bcorticosteroid\b",
    r"\bprednisolone\b", r"\bdexamethasone\b", r"\bbudesonide\b",
    r"\bmesalazine\b", r"\bsulfasalazine\b",
    r"\binfliximab\b", r"\badalimumab\b", r"\bvedolizumab\b",
    r"\bustekinumab\b", r"\btofacitinib\b",
    r"\bchemotherapy\b", r"\bimmunotherapy\b",
    r"\bcheckpoint inhibitor\b", r"\bpd.1\b", r"\bpd.l1\b",
    r"\bstatin\b", r"\batorvastatin\b", r"\brosuvastatin\b",
    r"\blevothyroxine\b", r"\bthyroid hormone\b",
    r"\boral contraceptive\b", r"\bhormone therapy\b",
    r"\bproton pump inhibitor\b",
    r"\blaxative\b", r"\bantidiarrheal\b",
    r"\brifaximin\b", r"\bneomycin\b",
    # ── Lifestyle ──
    r"\bexercise\b", r"\bphysical activity\b",
    r"\baerobic exercise\b", r"\bresistance training\b",
    r"\bsleep\b", r"\bstress\b", r"\bpsychological stress\b",
    r"\bsmoking\b", r"\btobacco\b", r"\balcohol\b",
    r"\bbreastfeeding\b", r"\bformula feeding\b",
    r"\bcesarean section\b", r"\bc.section\b", r"\bvaginal delivery\b",
]

DATASET_PATTERNS = [
    r"\bhuman microbiome project\b", r"\bhmp\b",
    r"\bcuratedmetagenomicdata\b", r"\bcuratedmetagenomic\b",
    r"\bfinrisk\b", r"\bamerica gut\b", r"\bamerican gut\b",
    r"\bbiome project\b", r"\bmetahit\b", r"\bmibc\b",
    r"\bgmrepo\b", r"\bmgx\b", r"\bgnps\b",
    r"\bncbi sra\b", r"\bsra\b", r"\bsequence read archive\b",
    r"\bena\b", r"\beuropean nucleotide archive\b",
    r"\bmgnify\b", r"\bimg.m\b",
    r"\bprjna\b", r"\berp\b", r"\bdrp\b",
    r"\bdbgap\b", r"\buk biobank\b",
    r"\bfinngen\b", r"\bbiome bank\b",
    r"\bmlvs\b", r"\bnhanes\b",
    r"\bpredimed\b", r"\bintermap\b",
    r"\bchild cohort\b", r"\bborn in bradford\b",
    r"\bgeneration r\b", r"\bviva\b",
    r"\becho cohort\b", r"\bcanue\b",
]

METABOLITE_PATTERNS = [
    r"\bscfa\b", r"\bshort.chain fatty acid", r"\bbutyrate\b", r"\bpropionate\b",
    r"\bacetate\b", r"\bsuccinate\b", r"\blactate\b", r"\bethanol\b",
    r"\bbile acid", r"\bsecondary bile acid", r"\bprimary bile acid",
    r"\bdeoxycholic acid\b", r"\bdca\b", r"\blithocholic acid\b", r"\blca\b",
    r"\bursodeoxycholic acid\b", r"\budca\b", r"\bchenodeoxycholic acid\b", r"\bcdca\b",
    r"\bindole.3.propionic acid\b", r"\bipa\b", r"\bindole.3.acetic acid\b", r"\biaa\b",
    r"\bindole\b", r"\btryptophan metabolite", r"\btryptamine\b", r"\bkynurenine\b",
    r"\bserotonin\b", r"\bgaba\b", r"\bdopamine\b",
    r"\blipopolysaccharide\b", r"\blps\b", r"\bpeptidoglycan\b", r"\bflagellin\b",
    r"\bteichoic acid\b", r"\bn.formylmethionine\b", r"\bfmet\b",
    r"\btrimethylamine\b", r"\btma\b", r"\btmao\b", r"\btrimethylamine n.oxide\b",
    r"\bhydrogen sulfide\b", r"\bh2s\b", r"\bmethane\b", r"\bhydrogen gas\b",
    r"\bformate\b", r"\bvalerate\b", r"\bisobutyrate\b", r"\bisovalerate\b",
    r"\bphenylacetate\b", r"\bp.cresol\b", r"\bindoxyl sulfate\b",
    r"\bsecondary metabolite", r"\bmicrobial metabolite",
]

GENE_PATTERNS = [
    r"\btlr4\b", r"\btlr2\b", r"\btlr5\b", r"\btlr9\b", r"\btlr\d\b",
    r"\bnod2\b", r"\bnod1\b", r"\bcard9\b", r"\bcard\d+\b",
    r"\bfxr\b", r"\bgpr41\b", r"\bgpr43\b", r"\bgpr109a\b",
    r"\bnf.kb\b", r"\bnfkb\b", r"\bnf.kappab\b",
    r"\bil.6\b", r"\bil.10\b", r"\bil.17\b", r"\bil.22\b", r"\bil.1b\b",
    r"\bil.18\b", r"\bil.23\b", r"\bil.12\b", r"\bil.4\b", r"\bil.13\b",
    r"\btnf.alpha\b", r"\btnf.a\b", r"\btnfa\b", r"\btnf\b",
    r"\bifn.gamma\b", r"\bifng\b", r"\bifn.y\b",
    r"\btgf.beta\b", r"\btgfb\b", r"\btgf.b\b",
    r"\bfoxp3\b", r"\brort\b", r"\brorgt\b", r"\bror.t\b",
    r"\bmuc2\b", r"\bmuc5ac\b", r"\bmucin\b",
    r"\boccludin\b", r"\bclaudin.1\b", r"\bzo.1\b", r"\btight junction protein",
    r"\breg3.?\b", r"\bdefensin\b", r"\bcathelicidin\b",
    r"\b16s rrna gene\b", r"\b16s rrna\b",
    r"\biga\b", r"\bsecretory iga\b", r"\bsiga\b",
    r"\bstat3\b", r"\bstat1\b", r"\bstat6\b",
    r"\bmyd88\b", r"\btrif\b", r"\btirap\b",
    r"\binflamasome\b", r"\bnlrp3\b",
]

PROTEIN_PATTERNS = [
    r"\bzonulin\b", r"\bcalprotectin\b", r"\blactoferrin\b", r"\blysozyme\b",
    r"\bmucin\b", r"\btight junction protein", r"\boccludin\b", r"\bclaudin\b",
    r"\bzo.1\b", r"\bzo.2\b", r"\bzo.3\b",
    r"\bpattern recognition receptor", r"\bprr\b",
    r"\btoll.like receptor", r"\bnod.like receptor",
    r"\bc.reactive protein\b", r"\bcrp\b",
    r"\blps.binding protein\b", r"\blbp\b",
    r"\bcd14\b", r"\bmyd88\b", r"\btrif\b",
    r"\bserum amyloid a\b", r"\bsaa\b",
    r"\bfibronectin\b", r"\blaminin\b", r"\bcollagen\b",
    r"\binterleukin\b", r"\bcytokine\b", r"\bchemokine\b",
    r"\badiponectin\b", r"\bleptin\b", r"\bghrelin\b",
    r"\bglp.1\b", r"\bglp.2\b", r"\bpyy\b", r"\bgip\b",
    r"\bsecretin\b", r"\bcholecystokinin\b", r"\bccl\d+\b", r"\bcxcl\d+\b",
]

BIOMARKER_PATTERNS = [
    r"\bfecal calprotectin\b", r"\bcalprotectin\b",
    r"\bserum zonulin\b", r"\bzonulin\b",
    r"\bc.reactive protein\b", r"\bcrp\b",
    r"\blps.binding protein\b", r"\blbp\b",
    r"\balpha diversity\b", r"\bshannon index\b", r"\bshannon diversity\b",
    r"\bsimpson index\b", r"\bchao1\b", r"\bfaith.s pd\b", r"\bfaith pd\b",
    r"\bobserved species\b", r"\bspecies richness\b",
    r"\bbeta diversity\b", r"\bbray.curtis\b", r"\bunifrac\b",
    r"\bweighted unifrac\b", r"\bunweighted unifrac\b", r"\bjaccard distance\b",
    r"\bfecal biomarker", r"\bserum biomarker", r"\bplasma biomarker",
    r"\binflammatory marker", r"\binflammatory biomarker",
    r"\bgut permeability marker", r"\bintestinal permeability marker",
    r"\bmicrobiome biomarker", r"\bmicrobial biomarker",
    r"\bfecal lactoferrin\b", r"\bfecal immunoglobulin\b",
    r"\bintestinal fatty acid binding protein\b", r"\bifabp\b",
    r"\bcitrulline\b", r"\bd.lactate\b",
]

PATHWAY_PATTERNS = [
    r"\btlr signaling\b", r"\btoll.like receptor signaling",
    r"\bnf.kb pathway\b", r"\bnfkb pathway\b", r"\bnf.kappab pathway\b",
    r"\bjak.stat pathway\b", r"\bjak.stat signaling\b",
    r"\bmapk pathway\b", r"\bmapk signaling\b",
    r"\bpi3k.akt pathway\b", r"\bpi3k.akt signaling\b",
    r"\bbutyrate metabolism\b", r"\bbutyrate production\b",
    r"\bbile acid metabolism\b", r"\bbile acid synthesis\b",
    r"\btryptophan metabolism\b", r"\btryptophan pathway\b",
    r"\bshort.chain fatty acid metabolism\b", r"\bscfa metabolism\b",
    r"\bscfa production\b", r"\bfermentation pathway\b",
    r"\bmucus degradation\b", r"\bmucin degradation\b",
    r"\bcolonization resistance\b",
    r"\bquorum sensing\b",
    r"\bhorizontal gene transfer\b", r"\bhgt\b",
    r"\bepithelial barrier function\b", r"\bgut barrier function\b",
    r"\bimmune signaling\b", r"\binnate immune pathway\b",
    r"\badaptive immune pathway\b", r"\bmucosal immunity\b",
    r"\binflammasome activation\b", r"\bnlrp3 activation\b",
    r"\bautophagy\b", r"\bapoptosis\b", r"\bpyroptosis\b",
    r"\bepigenetic regulation\b", r"\bhistone modification\b",
    r"\bkynurenine pathway\b", r"\bserotonin pathway\b",
]

POPULATION_PATTERNS = [
    r"\bhealthy adult", r"\bhealthy control", r"\bhealthy volunteer",
    r"\bibd patient", r"\bcrohn.s disease patient", r"\bulcerative colitis patient",
    r"\bobese individual", r"\bobese patient", r"\boverweight individual",
    r"\btype 2 diabetes patient", r"\bt2d patient",
    r"\bneonate\b", r"\bnewborn\b", r"\binfant\b", r"\bpremature infant\b",
    r"\belderly\b", r"\bolder adult", r"\baged individual",
    r"\bpregnant woman\b", r"\bpregnant women\b", r"\bpregnancy cohort\b",
    r"\bcancer patient", r"\bhiv patient", r"\bhiv.positive\b",
    r"\bantibiotic.treated\b", r"\bantibiotic.exposed\b",
    r"\bgerm.free mouse\b", r"\bgerm.free mice\b", r"\bgf mouse\b", r"\bgf mice\b",
    r"\bgnotobiotic mouse\b", r"\bgnotobiotic mice\b",
    r"\bhumanized mouse\b", r"\bhumanized mice\b",
    r"\bpediatric patient", r"\bchildren\b", r"\bchild cohort\b",
    r"\badolescent\b", r"\byoung adult\b",
    r"\bimmunocompromised\b", r"\bimmunosuppressed\b",
    r"\btransplant patient", r"\bstem cell transplant\b",
    r"\bcritically ill\b", r"\bicu patient\b",
]

DIETARY_COMPONENT_PATTERNS = [
    r"\bdietary fiber\b", r"\bfiber intake\b", r"\bfibre\b",
    r"\binulin\b", r"\bfructooligosaccharide", r"\bfos\b",
    r"\bgalactooligosaccharide", r"\bgos\b",
    r"\bresistant starch\b", r"\bpectin\b", r"\bbeta.glucan\b",
    r"\bpolyphenol\b", r"\bquercetin\b", r"\bresveratrol\b", r"\bcurcumin\b",
    r"\banthocyanin\b", r"\bflavonoid\b", r"\bphenolic compound",
    r"\bred meat\b", r"\bprocessed meat\b", r"\bprocessed food\b",
    r"\bhigh.fat diet\b", r"\bwestern diet\b",
    r"\bmediterranean diet\b", r"\bplant.based diet\b",
    r"\bfermented food\b", r"\bfermented beverage\b",
    r"\bwhole grain\b", r"\bvegetable\b", r"\bfruit\b",
    r"\blegume\b", r"\bpulse\b", r"\bnut\b",
    r"\bdairy\b", r"\bmilk\b", r"\byogurt\b", r"\bcheese\b",
    r"\bfish\b", r"\bseafood\b", r"\bomega.3\b",
    r"\bsoy\b", r"\bsoybean\b", r"\bisoflavone\b",
    r"\bgluten\b", r"\bwheat\b", r"\bbarley\b", r"\brye\b",
    r"\bsugar\b", r"\bsucrose\b", r"\bfructose\b", r"\bglucose\b",
    r"\bartificial sweetener\b", r"\bsaccharin\b", r"\baspartame\b",
    r"\bfood additive\b", r"\bemulsifier\b", r"\bpreservative\b",
]

IMMUNE_CELL_PATTERNS = [
    r"\bregulatory t cell", r"\btreg\b", r"\btregs\b",
    r"\bth17 cell", r"\bth17\b", r"\bt helper 17\b",
    r"\bth1 cell", r"\bth1\b", r"\bt helper 1\b",
    r"\bth2 cell", r"\bth2\b", r"\bt helper 2\b",
    r"\bdendritic cell", r"\bplasmacytoid dendritic\b",
    r"\bmacrophage\b", r"\bm1 macrophage\b", r"\bm2 macrophage\b",
    r"\bneutrophil\b", r"\bnatural killer cell", r"\bnk cell\b",
    r"\bilc3\b", r"\binnate lymphoid cell", r"\bilc\b",
    r"\bb cell\b", r"\bplasma cell\b", r"\bmemory b cell\b",
    r"\bmast cell\b", r"\beosinophil\b", r"\bbasophil\b",
    r"\bmonocyte\b", r"\bcd4\+ t cell\b", r"\bcd8\+ t cell\b",
    r"\bcytotoxic t cell\b", r"\beffector t cell\b",
    r"\blamina propria lymphocyte", r"\bintraepithelial lymphocyte",
    r"\bmucosal immune cell", r"\bgut immune cell",
    r"\bpaneth cell\b", r"\bgoblet cell\b", r"\benteroendocrine cell\b",
    r"\btuft cell\b", r"\bm cell\b",
]

CLINICAL_OUTCOME_PATTERNS = [
    r"\bremission\b", r"\bclinical remission\b", r"\bdeep remission\b",
    r"\brelapse\b", r"\bflare\b", r"\bdisease relapse\b",
    r"\bcolonization\b", r"\bengraftment\b", r"\bsuccessful engraftment\b",
    r"\bdysbiosis\b", r"\bmicrobial dysbiosis\b",
    r"\beubiosis\b", r"\bmicrobial eubiosis\b",
    r"\binflammation\b", r"\bsystemic inflammation\b", r"\bmucosal inflammation\b",
    r"\bintestinal permeability\b", r"\bleaky gut\b", r"\bgut permeability\b",
    r"\bbarrier function\b", r"\bgut barrier\b", r"\bepithelial barrier\b",
    r"\bmucosal healing\b", r"\bmucosal recovery\b",
    r"\bclinical response\b", r"\btreatment response\b",
    r"\bmicrobiome diversity\b", r"\bspecies richness\b",
    r"\bcommunity composition\b", r"\bmicrobial composition\b",
    r"\bweight loss\b", r"\bweight gain\b", r"\bbody weight change\b",
    r"\bglycemic control\b", r"\bblood glucose\b", r"\bhba1c\b",
    r"\blipid profile\b", r"\bcholesterol level\b",
    r"\bblood pressure\b", r"\bcardiovascular risk\b",
    r"\bcognitive function\b", r"\bbehavioral outcome\b",
    r"\bmortality\b", r"\bsurvival\b", r"\bhospitalization\b",
    r"\bsymptom score\b", r"\bquality of life\b", r"\bqol\b",
]

ENVIRONMENTAL_FACTOR_PATTERNS = [
    r"\bantibiotic exposure\b", r"\bantibiotic use\b", r"\bantibiotic treatment\b",
    r"\bbirth mode\b", r"\bmode of delivery\b",
    r"\bcesarean section\b", r"\bc.section\b", r"\bcesarean delivery\b",
    r"\bvaginal delivery\b", r"\bvaginal birth\b",
    r"\bbreastfeeding\b", r"\bbreast.fed\b", r"\bhuman milk\b",
    r"\bformula feeding\b", r"\bformula.fed\b", r"\binfant formula\b",
    r"\bgeographic location\b", r"\bgeography\b", r"\bcountry of origin\b",
    r"\bsocioeconomic status\b", r"\bses\b", r"\bincome level\b",
    r"\bsmoking\b", r"\btobacco use\b", r"\bcigarette\b",
    r"\balcohol consumption\b", r"\balcohol intake\b", r"\balcohol use\b",
    r"\bphysical activity\b", r"\bexercise habit\b", r"\bsedentary\b",
    r"\bstress\b", r"\bpsychological stress\b", r"\bchronic stress\b",
    r"\bsleep\b", r"\bsleep quality\b", r"\bsleep duration\b",
    r"\bage\b", r"\bsex\b", r"\bgender\b", r"\bbmi\b", r"\bbody mass index\b",
    r"\bethnic\b", r"\brace\b", r"\bpopulation background\b",
    r"\burban\b", r"\brural\b", r"\benvironment\b",
    r"\bpet ownership\b", r"\bfarm exposure\b", r"\bhygiene hypothesis\b",
    r"\bpollution\b", r"\bair quality\b", r"\bwater quality\b",
]

SEQUENCING_PLATFORM_PATTERNS = [
    r"\billumina miseq\b", r"\bmiseq\b",
    r"\billumina hiseq\b", r"\bhiseq\b",
    r"\billumina novaseq\b", r"\bnovaseq\b",
    r"\billumina nextseq\b", r"\bnextseq\b",
    r"\bpacbio sequel\b", r"\bpacbio\b", r"\bpacific biosciences\b",
    r"\boxford nanopore\b", r"\bnanopore minion\b", r"\bminion\b",
    r"\bnanopore\b", r"\bont\b",
    r"\bion torrent\b", r"\bion pgm\b", r"\bion proton\b",
    r"\b454 pyrosequencing\b", r"\b454 sequencing\b", r"\bpyrosequencing\b",
    r"\bsanger sequencing\b", r"\bsanger\b",
    r"\bpromethion\b", r"\bgridion\b",
    r"\bsequel ii\b", r"\bsequel iie\b",
]

OMICS_FEATURE_PATTERNS = [
    r"\botu\b", r"\botus\b", r"\boperational taxonomic unit",
    r"\basv\b", r"\basvs\b", r"\bamplicon sequence variant",
    r"\bmag\b", r"\bmags\b", r"\bmetagenome.assembled genome",
    r"\b16s rrna amplicon\b", r"\b16s amplicon\b",
    r"\bits amplicon\b", r"\bits sequencing\b",
    r"\bshotgun read\b", r"\bmetagenomic read\b",
    r"\bmetatranscript\b", r"\bmetatranscriptome\b",
    r"\bmetaprotein\b", r"\bmetaproteome\b",
    r"\bmetabolic profile\b", r"\bmetabolomic profile\b",
    r"\bfunctional gene\b", r"\bfunctional annotation\b",
    r"\bcog\b", r"\bcog category\b", r"\bclusters of orthologous\b",
    r"\bkegg ortholog\b", r"\bko\b", r"\bkegg pathway\b",
    r"\bpfam domain\b", r"\bpfam\b",
    r"\bgene catalog\b", r"\bgene cluster\b",
    r"\bcore genome\b", r"\bpan.genome\b", r"\baccessory genome\b",
    r"\bphylogenetic marker\b", r"\bmarker gene\b",
    r"\brelative abundance\b", r"\babundance profile\b",
    r"\btaxonomic profile\b", r"\bfunctional profile\b",
]

ENTITY_PATTERNS = {
    # ── Original 6 categories ─────────────────────────────────────────────────
    "taxon":                TAXA_PATTERNS,
    "disease":              DISEASE_PATTERNS,
    "method":               METHOD_PATTERNS,
    "body_site":            BODY_SITE_PATTERNS,
    "treatment":            TREATMENT_PATTERNS,
    "dataset":              DATASET_PATTERNS,
    # ── 12 new categories ─────────────────────────────────────────────────────
    "metabolite":           METABOLITE_PATTERNS,
    "gene":                 GENE_PATTERNS,
    "protein":              PROTEIN_PATTERNS,
    "biomarker":            BIOMARKER_PATTERNS,
    "pathway":              PATHWAY_PATTERNS,
    "population":           POPULATION_PATTERNS,
    "dietary_component":    DIETARY_COMPONENT_PATTERNS,
    "immune_cell":          IMMUNE_CELL_PATTERNS,
    "clinical_outcome":     CLINICAL_OUTCOME_PATTERNS,
    "environmental_factor": ENVIRONMENTAL_FACTOR_PATTERNS,
    "sequencing_platform":  SEQUENCING_PLATFORM_PATTERNS,
    "omics_feature":        OMICS_FEATURE_PATTERNS,
}

# ── Pre-compiled combined regex per category ──────────────────────────────────
# WHY PRE-COMPILE:
#   At 500K papers, calling re.finditer() on each of 300+ individual pattern
#   strings per paper = 150M+ regex operations. Combining patterns into one
#   compiled regex per category reduces this to 18 regex operations per paper
#   (one per entity category) — a 15-20× speedup for rule-based NER.
#
#   re.compile("|".join(patterns)) builds a single NFA that matches any of the
#   alternatives in one pass — same results, fraction of the cost.
#
# NOTE: Patterns are sorted longest-first so multi-word phrases (e.g.
#   "akkermansia muciniphila") are matched before single words ("akkermansia").
#   This prevents the single-word match from shadowing the species match.

def _compile_patterns(patterns: list) -> re.Pattern:
    """Combines a list of regex strings into one compiled pattern (longest first)."""
    sorted_pats = sorted(patterns, key=len, reverse=True)
    combined    = "|".join(f"(?:{p})" for p in sorted_pats)
    return re.compile(combined, re.IGNORECASE)


COMPILED_PATTERNS: dict = {
    label: _compile_patterns(pats)
    for label, pats in ENTITY_PATTERNS.items()
}


class NERExtractor:
    """
    Extracts named biomedical entities from paper text.
    Tier 1: Rule-based regex (always on)
    Tier 2: BioBERT NER model (optional, use_model=True)
    Tier 3: LLM extraction via Ollama (optional, use_llm=True)
    """

    def __init__(self, use_model: bool = False, use_llm: bool = False):
        self._model = None
        self._tokenizer = None
        self._model_loaded = False
        self._llm_extractor = None
        self._use_llm = use_llm

        if use_model:
            self._load_model()

        if use_llm:
            self._load_llm()

    def _load_model(self):
        try:
            import torch
            from transformers import pipeline as hf_pipeline

            # Use GPU if available — ~10× faster than CPU for BioBERT
            device = 0 if torch.cuda.is_available() else -1
            device_name = f"GPU (cuda:{device})" if device >= 0 else "CPU"

            logger.info(
                f"[NER] Loading BioBERT model on {device_name} "
                f"(first run downloads ~440MB)..."
            )
            self._model = hf_pipeline(
                "ner",
                model="d4data/biomedical-ner-all",
                aggregation_strategy="simple",
                device=device,
            )
            self._model_loaded = True
            logger.info(f"[NER] BioBERT model loaded on {device_name}")

        except ImportError:
            logger.warning(
                "[NER] transformers not installed — using rules only. "
                "Run: pip install transformers torch"
            )
        except Exception as e:
            logger.warning(f"[NER] Model load failed: {e} — using rules only")

    def _load_llm(self):
        try:
            import sys, os
            sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
            from semantic.llm_extractor import LLMExtractor
            self._llm_extractor = LLMExtractor()
            logger.info("[NER] LLM extractor (Ollama) loaded for Tier 3 extraction")
        except Exception as e:
            logger.warning(f"[NER] LLM extractor load failed: {e} — Tier 3 disabled")
            self._llm_extractor = None

    def extract(self, title: str, abstract: Optional[str], sections: list = None, full_text: str = None) -> List[NamedEntity]:
        title = title or ""
        abstract = abstract or ""
        base_text = f"{title} {abstract}".lower()
        base_text_original = f"{title} {abstract}"

        entities: List[NamedEntity] = []

        # Tier 1: Rule-based on title+abstract always
        # Tag title entities separately from abstract entities
        title_lower = title.lower()
        abstract_lower = abstract.lower()
        entities.extend(self._rule_based_extract(title_lower, source_section="title"))
        if abstract_lower.strip():
            entities.extend(self._rule_based_extract(abstract_lower, source_section="abstract"))

        # Tier 1: Also run on section content with per-section tagging
        if sections:
            ranked_sections = self._rank_sections(sections)
            for section in ranked_sections:
                section_type = getattr(section, 'section_type', None) or section.get('section_type', '')
                if section_type == 'abstract':
                    continue  # Already processed above
                content = getattr(section, 'content', None) or section.get('content', '')
                if content and content.strip():
                    entities.extend(self._rule_based_extract(
                        content.lower(),
                        source_section=section_type or "other",
                    ))

        # Tier 2: BioBERT — section-ranked chunked if sections available,
        # else chunked on full_text, else chunked on title+abstract
        if self._model_loaded and self._model:
            if sections:
                entities.extend(self._model_extract(base_text_original, sections=sections))
            elif full_text:
                entities.extend(self._model_extract(full_text))
            else:
                entities.extend(self._model_extract(base_text_original))

        # Tier 3: LLM — uses prioritized sections if available
        # Pass known entities from Tier 1+2 so LLM focuses on novel gaps
        if self._use_llm and self._llm_extractor:
            # Collect all entity names found so far by Tier 1 + Tier 2
            known = [e.text for e in entities]
            if sections:
                entities.extend(self._llm_extract(base_text_original, sections=sections, known_entities=known))
            elif full_text:
                entities.extend(self._llm_extract(full_text, known_entities=known))
            else:
                entities.extend(self._llm_extract(base_text_original, known_entities=known))

        # Deduplicate by (text, label)
        seen = set()
        unique = []
        for e in entities:
            key = (e.text.lower(), e.label)
            if key not in seen:
                seen.add(key)
                unique.append(e)

        return unique

    def _rule_based_extract(
        self, text_lower: str, source_section: Optional[str] = None
    ) -> List[NamedEntity]:
        results = []
        for label, compiled in COMPILED_PATTERNS.items():
            for match in compiled.finditer(text_lower):
                span = match.group(0).strip()

                if label == "taxon":
                    bad = {
                        "patients underwent", "data available",
                        "shotgun sequencing", "gut microbiome",
                        "human microbiome", "shotgun metagenomics",
                    }
                    if span.lower() in bad or len(span) < 2:
                        continue

                if label == "disease" and len(span) <= 3:
                    if not self._abbreviation_confirmed(span, match.start(), text_lower):
                        continue

                results.append(NamedEntity(
                    text=span,
                    label=label,
                    start=match.start(),
                    end=match.end(),
                    confidence=1.0,
                    source_section=source_section,
                ))
        return results

    # Context window for abbreviation confirmation (chars either side)
    _ABBREV_WINDOW = 80

    # For each ambiguous abbreviation: words that must appear nearby to confirm
    _ABBREV_CONTEXT: dict = {
        "uc":  ["colitis", "ulcerative", "ibd", "inflammatory", "bowel", "crohn"],
        "cd":  ["crohn", "disease", "ibd", "inflammatory", "bowel"],
        "ra":  ["arthritis", "rheumatoid", "joint", "synovial", "autoimmune"],
        "ms":  ["multiple sclerosis", "sclerosis", "demyelinating", "neurological",
                "relapsing"],
        "ibs": ["irritable", "bowel", "syndrome", "functional", "gastrointestinal"],
        "cfs": ["chronic", "fatigue", "syndrome", "me/cfs"],
        "bv":  ["bacterial", "vaginosis", "vaginal", "lactobacillus"],
        "uti": ["urinary", "tract", "infection", "cystitis", "urine"],
        "asd": ["autism", "spectrum", "disorder", "autistic", "neurodevelopmental"],
        "mdd": ["depressive", "depression", "major", "psychiatric"],
        "ald": ["alcoholic", "liver", "disease", "cirrhosis", "alcohol"],
        "hcc": ["hepatocellular", "carcinoma", "liver", "cancer", "hepatic"],
        "crc": ["colorectal", "cancer", "colon", "rectal", "adenocarcinoma"],
        "cdi": ["difficile", "clostridium", "clostridia", "infection", "diarrhea"],
    }

    def _abbreviation_confirmed(self, abbrev: str, pos: int, text: str) -> bool:
        """
        Returns True if the abbreviation at `pos` is confirmed by context.

        Looks for confirming words in a ±80 char window around the match.
        If the abbreviation is not in _ABBREV_CONTEXT, it's always accepted
        (only known-ambiguous ones need confirmation).
        """
        abbrev_lower = abbrev.lower()
        context_words = self._ABBREV_CONTEXT.get(abbrev_lower)
        if context_words is None:
            return True   # not a known-ambiguous abbreviation — accept as-is

        start  = max(0, pos - self._ABBREV_WINDOW)
        end    = min(len(text), pos + len(abbrev) + self._ABBREV_WINDOW)
        window = text[start:end].lower()

        return any(word in window for word in context_words)

    def _chunk_text(self, text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
        """Split text into overlapping word-based chunks."""
        words = text.split()
        if len(words) <= chunk_size:
            return [text]
        chunks = []
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunks.append(" ".join(words[start:end]))
            if end == len(words):
                break
            start += chunk_size - overlap
        return chunks

    def _rank_sections(self, sections: list) -> list:
        """Sort sections by importance (priority 1 first, skip priority 5)."""
        SECTION_PRIORITY = {
            # Priority 1 — highest scientific value
            "results": 1,
            "discussion": 1,
            # Priority 2 — important supporting content
            "data_availability": 2,
            "supplementary": 2,
            "statistical_analysis": 2,
            "bioinformatics": 2,
            "limitations": 2,
            "clinical_outcome": 2,
            # Priority 3 — methodology and context
            "methods": 3,
            "abstract": 3,
            "conclusion": 3,
            "study_population": 3,
            # Priority 4 — background context
            "introduction": 4,
            "background": 4,
            "other": 4,
            "future_directions": 4,
            "strengths": 4,
            # Priority 5 — skip (no scientific entity value)
            "references": 5,
            "acknowledgements": 5,
            "funding": 5,
            "conflict_of_interest": 5,
            "ethics": 5,
            "trial_registration": 5,
            "glossary": 5,
        }

        def get_priority(section):
            section_type = getattr(section, 'section_type', None) or section.get('section_type', 'other')
            return SECTION_PRIORITY.get(section_type.lower(), 4)

        ranked = sorted(sections, key=get_priority)
        # Filter out priority 5 sections (references, acknowledgements, etc.)
        return [s for s in ranked if get_priority(s) < 5]

    def _model_extract_chunks(
        self, text: str, max_chunks: int = 10,
        source_section: Optional[str] = None,
    ) -> List[NamedEntity]:
        """
        Run BioBERT on chunked text, deduplicate results.
        source_section is passed through to each entity for span tracking.
        """
        if not self._model or not text.strip():
            return []
        chunks = self._chunk_text(text, chunk_size=400, overlap=50)
        chunks = chunks[:max_chunks]
        all_entities = []
        for chunk in chunks:
            try:
                # Acquire GPU semaphore — prevents simultaneous BioBERT + Ollama on GPU
                with _GPU_SEMAPHORE:
                    raw_entities = self._model(chunk)
                for ent in raw_entities:
                    label = self._map_model_label(ent.get("entity_group", ""))
                    if label:
                        word = ent.get("word", "").strip()
                        # Skip WordPiece subword tokens and too-short spans
                        if not word or word.startswith("##") or len(word) < 2:
                            continue
                        # Skip tokens that are just punctuation or numbers
                        if word in {"-", ".", ",", "/", "(", ")", "[", "]"}:
                            continue
                        all_entities.append(NamedEntity(
                            text=word,
                            label=label,
                            confidence=round(float(ent.get("score", 0)), 3),
                            source_section=source_section,
                        ))
            except Exception as e:
                logger.warning(f"[NER] BioBERT chunk inference failed: {e}")
                continue

        # Deduplicate by (text.lower(), label)
        seen = set()
        unique = []
        for e in all_entities:
            key = (e.text.lower(), e.label)
            if key not in seen:
                seen.add(key)
                unique.append(e)
        return unique

    def _model_extract(self, text: str, sections: list = None) -> List[NamedEntity]:
        if not self._model or not text.strip():
            return []

        all_entities = []

        if sections:
            ranked_sections = self._rank_sections(sections)
            for section in ranked_sections:
                content      = getattr(section, 'content', None) or section.get('content', '')
                section_type = getattr(section, 'section_type', None) or section.get('section_type', 'other')
                if not content or not content.strip():
                    continue
                section_entities = self._model_extract_chunks(
                    content, max_chunks=10, source_section=section_type
                )
                all_entities.extend(section_entities)
                logger.debug(
                    f"[NER] BioBERT extracted {len(section_entities)} entities "
                    f"from {section_type} section"
                )
        else:
            all_entities = self._model_extract_chunks(text, max_chunks=20)

        return all_entities

    def _llm_extract(self, text: str, sections: list = None, known_entities: list = None) -> List[NamedEntity]:
        """Tier 3: LLM-based extraction via Ollama. Uses prioritized sections if available.
        
        known_entities: list of entity name strings already found by Tier 1+2.
                        Passed to the LLM prompt so it focuses on novel entities only.
        """
        if not self._llm_extractor or not text.strip():
            return []

        # If sections available, use results+discussion first, then others
        if sections:
            ranked = self._rank_sections(sections)
            # Concatenate top-priority sections up to 3000 chars
            priority_text = ""
            for section in ranked:
                content = getattr(section, 'content', None) or section.get('content', '')
                section_type = getattr(section, 'section_type', None) or section.get('section_type', 'unknown')
                if content:
                    priority_text += f"\n\n[SECTION: {section_type.upper()}]\n{content}"
                    if len(priority_text) >= 3000:
                        break
            extraction_text = priority_text[:3000] if priority_text.strip() else text[:3000]
        else:
            extraction_text = text[:3000]

        if not extraction_text.strip():
            return []

        try:
            # Acquire Ollama lock first (serialize LLM NER calls)
            # then GPU semaphore (prevent simultaneous BioBERT + Ollama on GPU)
            with _OLLAMA_NER_LOCK:
                with _GPU_SEMAPHORE:
                    candidate_entities, _ = self._llm_extractor.extract(
                        extraction_text,
                        known_entities=known_entities
                    )
            results = []
            label_map = {
                # Original 6
                "taxon": "taxon", "taxa": "taxon", "organism": "taxon",
                "bacteria": "taxon", "microbe": "taxon", "microorganism": "taxon",
                "disease": "disease", "condition": "disease", "disorder": "disease",
                "syndrome": "disease", "infection": "disease",
                "method": "method", "technique": "method", "assay": "method",
                "sequencing": "method", "analysis": "method",
                "body_site": "body_site", "tissue": "body_site", "organ": "body_site",
                "treatment": "treatment", "drug": "treatment", "therapy": "treatment",
                "intervention": "treatment", "supplement": "treatment",
                "dataset": "dataset", "database": "dataset", "cohort": "dataset",
                # New 12
                "metabolite": "metabolite", "metabolites": "metabolite",
                "scfa": "metabolite", "bile acid": "metabolite",
                "gene": "gene", "genes": "gene", "gene product": "gene",
                "receptor": "gene", "cytokine gene": "gene",
                "protein": "protein", "proteins": "protein",
                "biomarker": "biomarker", "biomarkers": "biomarker",
                "marker": "biomarker", "clinical marker": "biomarker",
                "pathway": "pathway", "pathways": "pathway",
                "signaling pathway": "pathway", "metabolic pathway": "pathway",
                "population": "population", "study population": "population",
                "patient group": "population", "subject": "population",
                "dietary component": "dietary_component",
                "dietary_component": "dietary_component",
                "food": "dietary_component", "nutrient": "dietary_component",
                "diet component": "dietary_component",
                "immune cell": "immune_cell", "immune_cell": "immune_cell",
                "cell type": "immune_cell", "lymphocyte": "immune_cell",
                "clinical outcome": "clinical_outcome",
                "clinical_outcome": "clinical_outcome",
                "outcome": "clinical_outcome", "endpoint": "clinical_outcome",
                "environmental factor": "environmental_factor",
                "environmental_factor": "environmental_factor",
                "exposure": "environmental_factor", "risk factor": "environmental_factor",
                "sequencing platform": "sequencing_platform",
                "sequencing_platform": "sequencing_platform",
                "platform": "sequencing_platform", "instrument": "sequencing_platform",
                "omics feature": "omics_feature", "omics_feature": "omics_feature",
                "feature": "omics_feature", "genomic feature": "omics_feature",
            }
            for ent in candidate_entities:
                if not ent.name or len(ent.name.strip()) < 2:
                    continue
                raw_type = (ent.entity_type or "").lower()
                label = label_map.get(raw_type)
                if not label:
                    # Try partial match
                    for key, mapped in label_map.items():
                        if key in raw_type:
                            label = mapped
                            break
                if label:
                    results.append(NamedEntity(
                        text=ent.name.strip(),
                        label=label,
                        confidence=0.75,
                    ))
            if results:
                logger.debug(f"[NER] LLM extracted {len(results)} additional entities")
            return results
        except Exception as e:
            logger.warning(f"[NER] LLM extraction failed: {e}")
            return []

    def _map_model_label(self, raw_label: str) -> Optional[str]:
        """
        Maps BioBERT entity group labels to our internal label vocabulary.

        Covers labels from:
          - d4data/biomedical-ner-all  (CRAFT/BioNLP scheme)
          - allenai/scibert            (generic scientific NER)
          - dmis-lab/biobert-large     (BC5CDR scheme)
          - any model using standard biomedical NER conventions
        """
        label_map = {
            # ── Taxa / Organisms ──────────────────────────────────────────────
            "Organism":                    "taxon",
            "Species":                     "taxon",
            "TAXON":                       "taxon",
            "B-SPECIES":                   "taxon",
            "Bacteria":                    "taxon",
            "Microorganism":               "taxon",
            # ── Diseases / Conditions ────────────────────────────────────────
            "Disease":                     "disease",
            "Disease_or_Phenotypic_Feature": "disease",
            "DISEASE":                     "disease",
            "Pathological_formation":      "disease",
            "Cancer":                      "disease",
            "DiseaseClass":                "disease",
            "B-DISEASE":                   "disease",
            # ── Chemicals / Metabolites ──────────────────────────────────────
            "Chemical":                    "metabolite",
            "Simple_chemical":             "metabolite",
            "CHEMICAL":                    "metabolite",
            "ChemicalEntity":              "metabolite",
            "Amino_acid":                  "metabolite",
            "B-CHEMICAL":                  "metabolite",
            "Drug":                        "treatment",
            "DrugClass":                   "treatment",
            # ── Genes / Proteins ─────────────────────────────────────────────
            "Gene_or_gene_product":        "gene",
            "GeneOrGeneProduct":           "gene",
            "GENE":                        "gene",
            "Gene":                        "gene",
            "B-GENE":                      "gene",
            "Protein":                     "protein",
            "ProteinFamily":               "protein",
            "B-PROTEIN":                   "protein",
            # ── Cell / Immune cells ──────────────────────────────────────────
            "Cell":                        "immune_cell",
            "Cell_type":                   "immune_cell",
            "CellLine":                    "immune_cell",
            "CellType":                    "immune_cell",
            "B-CELL_TYPE":                 "immune_cell",
            "Cellular_component":          "body_site",
            # ── Anatomy / Body sites ─────────────────────────────────────────
            "Anatomical_system":           "body_site",
            "Organ":                       "body_site",
            "Multi-tissue_structure":      "body_site",
            "Tissue":                      "body_site",
            "OrganismSubdivision":         "body_site",
            "AnatomicalEntity":            "body_site",
            "Developing_anatomical_structure": "body_site",
            "Immaterial_anatomical_entity":"body_site",
            "B-ANATOMY":                   "body_site",
            # ── Clinical outcomes / Phenotypes ───────────────────────────────
            "ClinicalTrial":               "clinical_outcome",
            "Phenotype":                   "clinical_outcome",
            "Measurement":                 "biomarker",
            # ── Methods / Techniques ─────────────────────────────────────────
            "ResearchTechnique":           "method",
            "LabTechnique":                "method",
            "Assay":                       "method",
            # ── Environment / Exposure ───────────────────────────────────────
            "EnvironmentalFactor":         "environmental_factor",
            "Exposure":                    "environmental_factor",
        }

        if not raw_label:
            return None

        # Exact match first (fastest)
        if raw_label in label_map:
            return label_map[raw_label]

        # Case-insensitive substring match as fallback
        raw_lower = raw_label.lower()
        for key, mapped in label_map.items():
            if key.lower() in raw_lower or raw_lower in key.lower():
                return mapped

        return None

    def group_entities(self, entities: List[NamedEntity]) -> dict:
        """
        Groups entities by label for easy access.

        Known labels (18 categories) go into their dedicated lists.
        Unknown labels — discovered by BioBERT or LLM but not in ENTITY_PATTERNS —
        go into 'other_entities' as a dict keyed by label type.
        This prevents novel entity types from being silently dropped.
        """
        groups: dict = {label: [] for label in ENTITY_PATTERNS}
        other: dict = {}  # open-world bucket for unknown types

        for ent in entities:
            if ent.label in groups:
                groups[ent.label].append(ent.text)
            elif ent.label and ent.label.strip():
                # Unknown entity type — store in open-world bucket
                label_key = ent.label.lower().strip().replace(" ", "_")
                if label_key not in other:
                    other[label_key] = []
                other[label_key].append(ent.text)

        # Deduplicate within each group
        result = {label: list(dict.fromkeys(items)) for label, items in groups.items()}

        # Deduplicate within other_entities
        result["other_entities"] = {
            k: list(dict.fromkeys(v)) for k, v in other.items()
        }

        return result
