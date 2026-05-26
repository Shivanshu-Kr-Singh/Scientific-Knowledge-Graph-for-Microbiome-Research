import re
def extract(text):
    text = text or ""
    out = {}
    n = re.search(r"(\d+)\s+"r"(patients|subjects|participants)",text,re.I)

    if n:
        out["sample_size"] = int(n.group(1))
    methods = []

    for m in [

        "16s",

        "16s rrna",

        "shotgun sequencing",

        "shotgun metagenomics",

        "metatranscriptomics",

        "whole genome sequencing"

    ]:

        if m in text.lower():
            methods.append(m)
    out["methods"] = methods
    
    dataset = re.findall(
    r"\b(?:PRJNA|SRP|GSE|ERP)\d+\b",text,re.I)
    out["datasets"] = dataset
    return out