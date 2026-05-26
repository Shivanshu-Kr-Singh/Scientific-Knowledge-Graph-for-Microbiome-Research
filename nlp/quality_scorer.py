def score(
    paper
):

    s = 0

    journal = paper.get(
        "journal_info"
    )

    quartile = None

    if journal:

        quartile = getattr(
            journal,
            "quartile",
            None
        )

    # Journal quality
    if quartile == "Q1":

        s += 0.3

    elif quartile == "Q2":

        s += 0.2

    # Study design

    study = paper.get(
        "study_design"
    )

    if study:

        stype = study.get(
            "type"
        )

        if stype == "rct":

            s += 0.3

        elif stype == "cohort":

            s += 0.2

    # Sample size

    sample = paper.get(
        "sample_size",
        0
    )

    if sample >= 500:

        s += 0.2

    elif sample >= 100:

        s += 0.1

    # Data availability

    if paper.get(
        "data_availability"
    ):

        s += 0.2

    return min(
        s,
        1.0
    )