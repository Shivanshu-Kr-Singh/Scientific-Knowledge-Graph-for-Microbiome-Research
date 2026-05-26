import requests

from bs4 import BeautifulSoup


class EuropePMCFullText:

    BASE = (

        "https://www.ebi.ac.uk/"

        "europepmc/webservices/rest/"

    )

    def fetch(
        self,
        pmcid
    ):

        url = (

            f"{self.BASE}"

            f"{pmcid}/fullTextXML"

        )

        r = requests.get(
            url,
            timeout=30
        )

        if r.status_code != 200:

            return None

        soup = BeautifulSoup(
            r.text,
            "xml"
        )

        def get_section(
            tag
        ):

            node = soup.find(
                tag
            )

            return (

                node.get_text(
                    " ",
                    strip=True
                )

                if node

                else ""

            )

        return {

            "abstract":

            get_section(
                "abstract"
            ),

            "methods":

            get_section(
                "methods"
            ),

            "results":

            get_section(
                "results"
            ),

            "discussion":

            get_section(
                "discussion"
            ),

            "fetch_source":

            "xml",

            "fetch_status":

            "success"
        }