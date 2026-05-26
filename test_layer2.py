from models import PaperRecord

from nlp.pipeline import NLPPipeline

paper = PaperRecord(

    title=
    "Shotgun metagenomics reveals gut microbiome changes in T2D",

    abstract=
    "200 patients underwent shotgun sequencing. Data available at PRJNA123456.",

)

pipe = NLPPipeline()

x = pipe.process_one(
    paper
)

print(
    x.model_dump()
)