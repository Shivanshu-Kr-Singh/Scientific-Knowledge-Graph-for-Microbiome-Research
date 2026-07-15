"""
Fast NER gap audit — checks which high-value biomedical terms
appear frequently in collected data but are absent from ner.py patterns.
"""
import json, re
from collections import Counter

with open('data/processed/collected_20260711_195131.json') as f:
    papers = json.load(f)

# Build a single string from all titles + abstracts
combined = ' '.join(
    (p.get('title') or '') + ' ' + (p.get('abstract') or '')
    for p in papers
).lower()

def count(term):
    return len(re.findall(r'\b' + re.escape(term) + r'\b', combined))

# ── Taxa missing from patterns ─────────────────────────────────────────────
taxa_candidates = [
    # New NCBI phylum names (2021-2024 reclassification)
    'pseudomonadota', 'bacillota', 'bacteroidota', 'actinomycetota',
    'fusobacteriota', 'spirochaetota', 'campylobacterota', 'desulfobacterota',
    # Genera commonly in microbiome papers
    'coprobacter', 'intestinibacter', 'intestinimonas', 'butyricoccus',
    'gemmiger', 'lachnoclostridium', 'sellimonas', 'anaerobutyricum',
    'phocea', 'lawsonibacter', 'agathobaculum', 'mediterraneibacter',
    'neglectibacter', 'herbinix', 'ruminiclostridium', 'hungateiclostridium',
    'acutalibacteraceae', 'muribaculaceae', 'tannerellaceae',
    # New Lactobacillaceae (Zheng 2020) — often not caught by old names
    'lactiplantibacillus plantarum', 'lacticaseibacillus rhamnosus',
    'limosilactobacillus reuteri', 'levilactobacillus brevis',
    'lentilactobacillus', 'companilactobacillus', 'secundilactobacillus',
    # Frequently cited species
    'bifidobacterium pseudocatenulatum', 'bifidobacterium catenulatum',
    'akkermansia muciniphila',  # should already be there
    'phocaeicola vulgatus', 'phocaeicola dorei',  # reclassified from Bacteroides
    'segatella copri',  # reclassified from Prevotella copri
    # Archaea
    'candidatus methanomethylophilales', 'methanomassiliicoccus',
    # Gut virome
    'crass-like phage', 'crassphage', 'microviridae', 'siphoviridae',
]

print("=== TAXA (count in data) ===")
for t in taxa_candidates:
    c = count(t)
    if c > 0:
        print(f"  {c:4d}  {t}")

# ── Diseases missing ────────────────────────────────────────────────────────
disease_candidates = [
    'long covid', 'post-covid', 'post-acute sequelae', 'pasc',
    'masld', 'metabolic dysfunction-associated',
    'pots', 'postural orthostatic tachycardia',
    'primary sclerosing cholangitis', 'psc',
    'primary biliary cholangitis', 'pbc',
    'microscopic colitis', 'collagenous colitis', 'lymphocytic colitis',
    'diverticulosis', 'diverticular disease',
    'functional dyspepsia',
    'eosinophilic esophagitis', 'eoe',
    'small for gestational age', 'intrauterine growth restriction',
    'childhood asthma', 'pediatric asthma',
    'food protein-induced', 'fpies',
    'mast cell activation', 'mcas',
    'dysautonomia',
    'long-term covid', 'covid long hauler',
    'acute pancreatitis', 'chronic pancreatitis',
    'autoimmune hepatitis',
    'wilson disease',
    'hereditary hemochromatosis',
    'primary hyperoxaluria',
    'hyperthyroidism', 'hypothyroidism',
]

print("\n=== DISEASES (count in data) ===")
for d in disease_candidates:
    c = count(d)
    if c > 0:
        print(f"  {c:4d}  {d}")

# ── Methods / tools missing ─────────────────────────────────────────────────
method_candidates = [
    # Bioinformatics tools not in patterns
    'ancom', 'ancom-bc', 'aldex2', 'songbird', 'feast',
    'sourcetracker', 'sourcetracker2', 'woltka',
    'tax4fun2', 'phyloglm', 'gneiss', 'gemelli',
    'mmvec', 'sparcc', 'spiec-easi',
    'picrust', 'picrust2',  # check if already matched
    'taxize', 'dada2',  # check
    'metagenomics-assembled', 'mag reconstruction', 'binning',
    'checkm2', 'gtdb-tk', 'gtdbtk',  # check
    'anvi\'o', 'anvio',
    'metaspades', 'megahit',  # check
    'minimap2', 'flye', 'medaka',
    'hifiasm', 'canu',
    'metabinner', 'semibin', 'semibin2', 'vamb',
    'deepvirfinder', 'virsorter', 'virsorter2',
    'vcontact2', 'iphop',
    'vibrant',
    'metawrap',
    'drep', 'd-rep',
    'eggnog-mapper', 'eggnog mapper',
    'interproscan',
    'rgi', 'resfinder', 'amrfinder',
    'mlst', 'multi-locus sequence typing',
    'whole genome alignment',
    'pangenome', 'pan-genome',  # check
    'phyloflash',
    'silva database', 'silva', 'greengenes', 'greengenes2',
    'gtdb', 'ncbi taxonomy',
    'unifrac',  # check
    # Statistical methods
    'linear mixed model', 'mixed-effects model',
    'negative binomial', 'zero-inflated',
    'dirichlet-multinomial', 'dirichlet multinomial',
    'compositional data analysis', 'coda',
    'principal coordinate analysis', 'pcoa',
    'principal component analysis', 'pca',
    'redundancy analysis', 'rda',
    'constrained ordination',
    'network analysis', 'co-occurrence network',
    'machine learning', 'deep learning', 'neural network',
    'xgboost', 'gradient boosting',
    'logistic regression',
    'survival analysis', 'cox regression',
    # Wet lab methods
    'metagenome-assembled genome',
    'culturomics', 'high-throughput culturomics',
    'germ-free', 'gnotobiotic',  # check
    'colonoid', 'enteroid', 'organoid',
    'gut-on-a-chip', 'organ-on-a-chip',
    'tnbs', 'dss colitis', 'dss model',
]

print("\n=== METHODS (count in data) ===")
for m in method_candidates:
    c = count(m)
    if c > 0:
        print(f"  {c:4d}  {m}")

# ── Metabolites missing ────────────────────────────────────────────────────
metabolite_candidates = [
    'urolithin', 'urolithin a', 'urolithin b',
    'equol', 's-equol',
    'enterodiol', 'enterolactone',
    'ellagic acid', 'ellagitannin', 'punicalagin',
    'phenylpropionic acid', 'phenylacetic acid',
    'hippuric acid',
    'dimethyl sulfoxide', 'dmso',
    'hydrogen peroxide',
    'nitric oxide',
    'histamine',
    'agmatine',
    'cadaverine', 'putrescine', 'spermidine', 'spermine',
    'polyamine',
    'n-acyl homoserine lactone', 'quorum sensing molecule',
    'bacteriocin',
    'reuterin', '3-hpa',
    'phenol', 'p-cresol', 'cresol',
    'skatole', 'indoxyl',
    'secondary metabolite',
    'fermentation product',
    'acetaldehyde',
    'oxalate',
    'oxalic acid',
    'mucin-degrading',
]

print("\n=== METABOLITES (count in data) ===")
for m in metabolite_candidates:
    c = count(m)
    if c > 0:
        print(f"  {c:4d}  {m}")

# ── Datasets missing ──────────────────────────────────────────────────────
dataset_candidates = [
    'ibdmdb', 'ibdmdb2',
    'lifelines', 'lifelines deep',
    'rotterdam study',
    'nhanes',  # check
    'uk biobank',  # check
    'finngen',
    'million veteran program', 'mvp',
    'all of us',
    'metabolon',
    'mgnify',  # check
    'ibdmdb', 'igram', 'nicu',
    'mibioome', 'diabimmune',
    'copsac', 'child', 'echo cohort',
    'prism cohort', 'sparc ibd',
    'icoast', 'risk',
    'biobank japan',
    'estonian biobank',
    'qatar biobank',
    'metagenomics of the human intestinal tract', 'metahit',
    'gmrepo',  # check
    'microbiota vault',
    'guthealthdb', 'gutmgene',
    'curatedmetagenomic', 'curatedmetagenomicdata',  # check
    'mgx.hmpdacc',
    'sra accession', 'bioproject',
    'ncbi bioproject',
    'ebi metagenomics',
]

print("\n=== DATASETS (count in data) ===")
for d in dataset_candidates:
    c = count(d)
    if c > 0:
        print(f"  {c:4d}  {d}")

print("\nDone.")
