import json
from pathlib import Path
from threading import Lock

AUDIT_DIR = (
    Path(__file__)
    .parent.parent
    / "data"
    / "audit"
)

_LOCK = Lock()


class AuditLogger:

    MAP = {

        "keep":
        AUDIT_DIR / "kept.json",

        "reject":
        AUDIT_DIR / "rejected.json",

        "review":
        AUDIT_DIR / "review.json",

        "llm":
        AUDIT_DIR / "llm_verified.json"
    }

    @classmethod
    def log(
        cls,
        paper,
        verdict
    ):

        decision = (
            "review"
            if verdict.review
            else (
                "keep"
                if verdict.keep
                else "reject"
            )
        )

        path = cls.MAP[
            decision]

        item = {
            "title":
            paper.title,

            "source":
            paper.source,

            "year":
                getattr(paper, "publication_year", getattr(paper, "pub_year", None)),

            "decision":
            decision,

            "stage":
            verdict.stage,

            "score":
            verdict.score,

            "reason":
            verdict.reason}

        cls._append(
            path,
            item
        )

    @classmethod
    def log_llm(cls,paper,verdict):
        item = {

            "title":
            paper.title,

            "source":
            paper.source,

            "keep":
            verdict.keep,

            "confidence":
            verdict.confidence,

            "reason":
            verdict.reason,

            "cached":
            verdict.cached
        }

        cls._append(
            cls.MAP["llm"],
            item
        )

    @classmethod
    def _append(
        cls,
        path,
        item
    ):

        with _LOCK:

            path.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            data = []

            if path.exists():

                try:

                    with open(
                        path
                    ) as f:

                        data = json.load(
                            f
                        )

                except:

                    data = []

            data.append(
                item
            )

            with open(
                path,
                "w"
            ) as f:

                json.dump(
                    data,
                    f,
                    indent=2
                )