"""
Find real papers that were KEPT and REJECTED by Stage 3.5 embedding filter,
and show which stored paper they were most similar to.
"""
import json
import numpy as np
from pathlib import Path

BASE = Path(".")
AUDIT = BASE / "data" / "audit"
EMB = BASE / "data" / "embeddings"

# Load store
pos_vecs = np.load(EMB / "positive.npy")
neg_vecs = np.load(EMB / "negative.npy")
pos_meta = json.load(open(EMB / "positive_meta.json"))
neg_meta = json.load(open(EMB / "negative_meta.json"))

# Load audit data
kept = json.load(open(AUDIT / "kept.json"))
rej = json.load(open(AUDIT / "rejected.json"))
all_recs = kept + rej

# Find papers decided at stage3_ml (they went through embedding)
# Stage 3 ML papers are ones that passed through Stage 2 as borderline
# For embedding examples, let's find papers with high/low ML probabilities
# that also have abstracts for good embedding

# Load embedding model
from collectors.embedding_model import EmbeddingModel
model = EmbeddingModel()

def cosine_sim(a, b):
    a_norm = a / (np.linalg.norm(a) + 1e-10)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return np.dot(b_norm, a_norm)

# Pick papers that were KEPT by ML (high prob) and REJECTED by ML (low prob)
# These went through the embedding path
s3_kept = [x for x in kept if x['stage'] == 'stage3_ml']
s3_rej = [x for x in rej if x['stage'] == 'stage3_ml']

print("=" * 80)
print("PAPERS ACCEPTED (Stage 3 ML KEEP) — showing similarity to store")
print("=" * 80)

for paper in s3_kept[:3]:
    title = paper.get('title', '')
    abstract = paper.get('abstract', '')
    doi = paper.get('doi', '')
    
    vec = model.encode_paper(title, abstract)
    
    pos_sims = cosine_sim(vec, pos_vecs)
    top_pos_idx = int(np.argmax(pos_sims))
    top_pos_sim = float(pos_sims[top_pos_idx])
    top_pos = pos_meta[top_pos_idx]
    
    neg_sims = cosine_sim(vec, neg_vecs)
    top_neg_idx = int(np.argmax(neg_sims))
    top_neg_sim = float(neg_sims[top_neg_idx])
    top_neg = neg_meta[top_neg_idx]
    
    print(f"\nPaper: {title[:90]}")
    print(f"  DOI: {doi}")
    print(f"  ML prob: {paper['score']:.4f}")
    print(f"  Closest POSITIVE (sim={top_pos_sim:.4f}):")
    print(f"    Title: {top_pos.get('title','')[:85]}")
    print(f"    DOI: {top_pos.get('doi','n/a')}")
    print(f"  Closest NEGATIVE (sim={top_neg_sim:.4f}):")
    print(f"    Title: {top_neg.get('title','')[:85]}")
    print(f"    DOI: {top_neg.get('doi','n/a')}")
    print()

print("\n" + "=" * 80)
print("PAPERS REJECTED (Stage 3 ML REJECT) — showing similarity to store")
print("=" * 80)

for paper in s3_rej[:3]:
    title = paper.get('title', '')
    abstract = paper.get('abstract', '')
    doi = paper.get('doi', '')
    
    vec = model.encode_paper(title, abstract)
    
    pos_sims = cosine_sim(vec, pos_vecs)
    top_pos_idx = int(np.argmax(pos_sims))
    top_pos_sim = float(pos_sims[top_pos_idx])
    top_pos = pos_meta[top_pos_idx]
    
    neg_sims = cosine_sim(vec, neg_vecs)
    top_neg_idx = int(np.argmax(neg_sims))
    top_neg_sim = float(neg_sims[top_neg_idx])
    top_neg = neg_meta[top_neg_idx]
    
    print(f"\nPaper: {title[:90]}")
    print(f"  DOI: {doi}")
    print(f"  ML prob: {paper['score']:.4f}")
    print(f"  Closest POSITIVE (sim={top_pos_sim:.4f}):")
    print(f"    Title: {top_pos.get('title','')[:85]}")
    print(f"    DOI: {top_pos.get('doi','n/a')}")
    print(f"  Closest NEGATIVE (sim={top_neg_sim:.4f}):")
    print(f"    Title: {top_neg.get('title','')[:85]}")
    print(f"    DOI: {top_neg.get('doi','n/a')}")
    print()
