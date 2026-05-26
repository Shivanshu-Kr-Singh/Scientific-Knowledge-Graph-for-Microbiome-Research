from nlp.fulltext.europepmc_fulltext import EuropePMCFullText
from nlp.fulltext.pdf_parser import PDFParser
from nlp.fulltext.web_scraper import WebScraper


class FullTextOrchestrator:

    def __init__(self):
        self.xml = EuropePMCFullText()
        self.pdf = PDFParser()
        self.web = WebScraper()

    def fetch(self,paper):
        if getattr(paper,"pmcid",None):
            record = self.xml.fetch(paper.pmcid)

            if record:
                return record

        if getattr(paper,"pdf_url",None):
            record = self.pdf.fetch(paper.pdf_url)

            if record:
                return record

        if getattr(paper,"full_text_url",None):
            return self.web.fetch(paper.full_text_url)

        return None