from src.basic_classes.sentence import Sentence
from src.basic_classes.entity import Entity
from src.basic_classes.summary import Summary


class Document:

    def __init__(self, id_document: int = -1, name: str = None, title: str = None, text: str = None,
                 reference_summary: str = None):
        self.id_document = id_document
        self.name = name
        self.title = title
        self.tokens_title = []
        self.text = text
        self.sentences = []
        self.entities = []
        self.tokens = []
        self.tokens_no_stopwords = []
        self.concepts = []
        self.references_summaries = None
        self.reference_summary = reference_summary
        self.candidate_summaries = []

    def add_sentence(self, sentence: Sentence):
        self.sentences.append(sentence)

    def add_tokens(self, tokens: list):
        self.tokens.extend(tokens)

    def add_tokens_no_stopwords(self, tokens_no_stopwords: list):
        self.tokens_no_stopwords.extend(tokens_no_stopwords)

    def add_entity(self, entity: Entity):
        self.entities.append(entity)

    def add_candidate_summary(self, summary: Summary):
        self.candidate_summaries.append(summary)
