import fitz
import tempfile
import requests


class PDFParser:

    def fetch(
        self,
        url
    ):

        r = requests.get(
            url,
            timeout=60
        )

        if r.status_code != 200:

            return None

        tmp = tempfile.NamedTemporaryFile(
            suffix=".pdf"
        )

        tmp.write(
            r.content
        )

        doc = fitz.open(
            tmp.name
        )

        text = ""

        for page in doc:

            text += (
                page.get_text()
                + "\n"
            )

        return {

            "full_text":
            text,

            "fetch_source":
            "pdf",

            "fetch_status":
            "success"
        }