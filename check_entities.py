import json

with open('data/processed/enriched_20260607_195836.json', encoding='utf-8') as f:
    papers = json.load(f)

print(f'Total papers: {len(papers)}')
print()

total_entities = 0
for i, p in enumerate(papers[:8]):
    title = p.get('title', '')[:65]
    entities = p.get('entities', [])
    taxa = p.get('taxa', [])
    diseases = p.get('diseases', [])
    methods = p.get('methods', [])
    metabolites = p.get('metabolites', [])
    total_entities += len(entities)
    print(f"Paper {i+1}: {title}")
    print(f"  Total entities: {len(entities)}")
    print(f"  Taxa ({len(taxa)}): {taxa[:3]}")
    print(f"  Diseases ({len(diseases)}): {diseases[:3]}")
    print(f"  Methods ({len(methods)}): {methods[:3]}")
    print(f"  Metabolites ({len(metabolites)}): {metabolites[:3]}")
    # Show extraction methods used per entity
    extraction_methods = set()
    for e in entities:
        src = e.get('grounding_source', 'unknown')
        extraction_methods.add(src)
    print(f"  Grounding sources used: {extraction_methods}")
    print()

print(f'Avg entities per paper (first 8): {round(total_entities/min(8,len(papers)), 1)}')
