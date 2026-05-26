import re


PATTERNS = {

    "rct":
    r"randomized|trial",

    "cohort":
    r"(?:cohort|\d+\s+(?:patients|subjects|participants)\b)",

    "case_control":
    r"case.control",

    "cross_sectional":
    r"cross.sectional",

    "pilot":
    r"pilot",

    "protocol":
    r"protocol",

    "observational":
    r"\benrolled\b|\brecruited\b"

}


def extract_design(text):
    text = (text or "").lower()
    for k, v in PATTERNS.items():
        if re.search(v,text):
            return {"type": k, "confidence": 0.9}
    return None