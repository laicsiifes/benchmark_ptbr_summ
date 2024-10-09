class Token:

    def __init__(self, id_token: int = -1, text: str = None, lemma: str = None, pos: str = None, token_type: str = None,
                 is_stopword: bool = False):
        self.id_token = id_token
        self.text = text
        self.lemma = lemma
        self.pos = pos
        self.token_type = token_type
        self.is_stopword = is_stopword
        self.sub_tokens = []
        self.weight = 0
        self.weights = {}

    def add_sub_token(self, token):
        self.sub_tokens.append(token)

    def add_weight(self, name, value):
        self.weights[name] = value

    def __repr__(self):
        return self.text

    def __hash__(self):
        return hash(self.lemma)

    def __eq__(self, other):
        return isinstance(other, Token) and other.lemma == self.lemma
