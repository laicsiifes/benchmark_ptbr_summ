from src.basic_classes.token_1 import Token


class Sentence:

    def __init__(self):
        self.full_id_sentence: str = ''
        self.id_sentence: int = -1
        self.full_id: str = ''
        self.text: str = ''
        self.tokens: list = []
        self.tokens_no_stopwords: list = []
        self.concepts: list = []
        self.entities: list = []
        self.scores: dict = {}
        self.relevance_score: float = 0.0
        self.is_removed: bool = False

    def add_token(self, token: Token):
        self.tokens.append(token)

    def add_token_no_stopword(self, token: Token):
        self.tokens_no_stopwords.append(token)

    def add_score(self, method_name: str, score_value: float):
        self.scores[method_name] = score_value

    def __repr__(self):
        return self.text
