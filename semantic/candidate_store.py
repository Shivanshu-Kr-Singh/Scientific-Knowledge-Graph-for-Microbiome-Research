from pydantic import BaseModel
from typing import Optional

class CandidateEntity(BaseModel):
    name: str
    entity_type: str
    canonical: str | None = None
    ontology: str | None = None
    ontology_id: str | None = None
    grounded: bool = False

class CandidateRelation(BaseModel):
    subject: str
    predicate: str
    object: str
    confidence: float = 0.8