def rank(paper):
    score = (paper.quality_score)
    score += min(paper.evidence_score/ 1000,0.2)

    return round(min(score, 1.0),3)