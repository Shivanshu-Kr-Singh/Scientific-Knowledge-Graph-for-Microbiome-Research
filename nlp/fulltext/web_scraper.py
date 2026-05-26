import trafilatura


class WebScraper:

    def fetch(
        self,
        url
    ):

        downloaded = (
            trafilatura.fetch_url(
                url
            )
        )

        text = (
            trafilatura.extract(
                downloaded
            )
        )

        if not text:

            return None

        return {

            "full_text":
            text,

            "fetch_source":
            "web",

            "fetch_status":
            "success"
        }