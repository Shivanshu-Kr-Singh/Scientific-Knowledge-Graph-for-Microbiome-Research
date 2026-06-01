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

from nlp.enriched_record import NamedEntity


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
            from transformers import pipeline
            logger.info("[NER] Loading BioBERT model (first run downloads ~440MB)...")
            self._model = pipeline(
                "ner",
                model="d4data/biomedical-ner-all",
                aggregation_strategy="simple",
                device=-1,
            )
            self._model_loaded = True
            logger.info("[NER] BioBERT model loaded")
        except ImportError:
            logger.warning("[NER] transformers not installed — using rules only. Run: pip install transformers torch")
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

    def extract(self, title: str, abstract: Optional[str]) -> List[NamedEntity]:
        title = title or ""
        abstract = abstract or ""
        text = f"{title} {abstract}".lower()
        full_text = f"{title} {abstract}"

        entities: List[NamedEntity] = []

        # Tier 1: Rule-based
        entities.extend(self._rule_based_extract(text))

        # Tier 2: BioBERT
        if self._model_loaded and self._model:
            entities.extend(self._model_extract(full_text))

        # Tier 3: LLM (Ollama) — runs on all papers to catch novel entities
        # missed by regex and BioBERT. Results are cached so re-runs are instant.
        if self._use_llm and self._llm_extractor:
            entities.extend(self._llm_extract(full_text))

        # Deduplicate by (text, label)
        seen = set()
        unique = []
        for e in entities:
            key = (e.text.lower(), e.label)
            if key not in seen:
                seen.add(key)
                unique.append(e)

        return unique

    def _rule_based_extract(self, text_lower: str) -> List[NamedEntity]:
        results = []
        for label, patterns in ENTITY_PATTERNS.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text_lower, re.IGNORECASE):
                    span = match.group(0).strip()
                    if label == "taxon":
                        bad = {
                            "patients underwent", "data available",
                            "shotgun sequencing", "gut microbiome",
                            "human microbiome", "shotgun metagenomics",
                        }
                        if span.lower() in bad:
                            continue
                    if len(span) < 2:
                        continue
                    results.append(NamedEntity(
                        text=span,
                        label=label,
                        start=match.start(),
                        end=match.end(),
                        confidence=1.0,
                    ))
        return results

    def _model_extract(self, text: str) -> List[NamedEntity]:
        if not self._model or not text.strip():
            return []
        try:
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

    def _llm_extract(self, text: str) -> List[NamedEntity]:
        """Tier 3: LLM-based extraction via Ollama."""
        if not self._llm_extractor or not text.strip():
            return []
        try:
            candidate_entities, _ = self._llm_extractor.extract(text)
            results = []
            label_map = {
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
        label_map = {
            "Chemical":             "treatment",
            "Simple_chemical":      "treatment",
            "Disease":              "disease",
            "Gene_or_gene_product": None,
            "Organism":             "taxon",
            "Species":              "taxon",
            "Cell":                 "body_site",
            "Protein":              None,
        }
        for key, mapped in label_map.items():
            if key.lower() in raw_label.lower():
                return mapped
        return None

    def group_entities(self, entities: List[NamedEntity]) -> dict:
        groups: dict = {label: [] for label in ENTITY_PATTERNS}
        for ent in entities:
            if ent.label in groups:
                groups[ent.label].append(ent.text)
        return {label: list(dict.fromkeys(items)) for label, items in groups.items()}
