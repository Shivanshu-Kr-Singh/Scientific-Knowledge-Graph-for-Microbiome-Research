from semantic.llm_extractor import (LLMExtractor)

x = LLMExtractor()

ents, rels = x.extract(

"""
500 T2D patients
received Vitamin D.

Akkermansia increased.

Butyrate improved.

miR-21 changed.
"""

)

print(ents)

print(rels)