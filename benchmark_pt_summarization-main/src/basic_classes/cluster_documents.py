from src.basic_classes.document import Document
from src.basic_classes.summary import Summary


class ClusterDocuments:

    def __init__(self, name: str):
        self.name = name
        self.candidate_summaries = []
        self.extractive_reference_summaries = []
        self.abstractive_reference_summaries = []
        self.documents = []
        self.sentences = []
        self.concepts = []

    def add_candidate_summary(self, summary: Summary):
        self.candidate_summaries.append(summary)

    def add_extractive_reference_summary(self, reference_summary: str):
        self.extractive_reference_summaries.append(reference_summary)

    def add_abstractive_reference_summary(self, reference_summary: str):
        self.abstractive_reference_summaries.append(reference_summary)

    def add_document(self, document: Document):
        self.documents.append(document)

    def add_sentences(self, sentences: list):
        self.sentences.extend(sentences)
