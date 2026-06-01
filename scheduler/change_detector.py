import hashlib

def compute_hash(paper):
    txt = " ".join([
        str(paper.title or ""),
        str(paper.abstract or ""),
        str(paper.mesh_terms or "")
    ])
    return hashlib.md5(txt.encode()).hexdigest()