import json, re
with open('data/processed/collected_20260711_195131.json') as f:
    papers = json.load(f)
txt = ' '.join(((p.get('title') or '') + ' ' + (p.get('abstract') or '')).lower() for p in papers)
checks = [
    ('TAXA','phocaeicola vulgatus'),('TAXA','segatella copri'),
    ('TAXA','gemmiger'),('TAXA','lachnoclostridium'),('TAXA','muribaculaceae'),
    ('TAXA','lentilactobacillus'),('TAXA','pseudomonadota'),('TAXA','bacillota'),
    ('TAXA','actinomycetota'),('TAXA','bacteroidota'),('TAXA','anaerobutyricum'),
    ('TAXA','sellimonas'),('TAXA','candidatus'),('TAXA','phocaeicola'),
    ('DIS','long covid'),('DIS','post-covid'),('DIS','masld'),
    ('DIS','primary sclerosing cholangitis'),('DIS','microscopic colitis'),
    ('DIS','functional dyspepsia'),('DIS','eosinophilic esophagitis'),
    ('DIS','acute pancreatitis'),('DIS','autoimmune hepatitis'),
    ('DIS','diverticulosis'),('DIS','diverticular disease'),
    ('DIS','pots'),('DIS','postural orthostatic'),
    ('MET','ancom-bc'),('MET','aldex2'),('MET','songbird'),('MET','woltka'),
    ('MET','sourcetracker'),('MET','checkm2'),('MET','minimap2'),('MET','semibin'),
    ('MET','virsorter'),('MET','deepvirfinder'),('MET','eggnog'),
    ('MET','resfinder'),('MET','culturomics'),('MET','organoid'),
    ('MET','dss colitis'),('MET','greengenes2'),('MET','silva'),
    ('MET','pcoa'),('MET','co-occurrence network'),('MET','network analysis'),
    ('METAB','urolithin'),('METAB','equol'),('METAB','enterolactone'),
    ('METAB','hippuric'),('METAB','spermidine'),('METAB','putrescine'),
    ('METAB','bacteriocin'),('METAB','p-cresol'),('METAB','skatole'),
    ('METAB','oxalate'),('METAB','polyamine'),
    ('DATA','ibdmdb'),('DATA','lifelines'),('DATA','diabimmune'),
    ('DATA','bioproject'),('DATA','ncbi bioproject'),('DATA','gutmgene'),
]
by_cat = {}
for cat, term in checks:
    c = txt.count(term)
    if c > 0:
        by_cat.setdefault(cat, []).append((c, term))
for cat in ['TAXA','DIS','MET','METAB','DATA']:
    items = sorted(by_cat.get(cat, []), reverse=True)
    if items:
        print(f"\n=== {cat} ===")
        for c, t in items:
            print(f"  {c:4d}  {t}")
