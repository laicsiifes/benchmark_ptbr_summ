import spacy
from unicodedata import normalize

from src.basic_classes.document import Document
from src.basic_classes.sentence import Sentence
from src.basic_classes.sentence import Token

nlp = spacy.load('pt_core_news_sm')


def process_document(document: Document):

    doc = nlp(document.text)

    document.sentences = []
    document.tokens_no_stopwords = []

    for id_sent, sent_spacy in enumerate(doc.sents, start=1):

        sentence = Sentence()

        sentence.id_sentence = id_sent
        sentence.text = sent_spacy.text

        for id_token, token_spacy in enumerate(sent_spacy, start=1):

            token_type = 'WORD'

            if token_spacy.pos_ in ['PUNCT', 'SPACE']:
                token_type = 'PUNCT'
            elif token_spacy.pos_ == 'NUM':
                token_type = 'NUM'
            elif token_spacy.pos_ == 'SYM':
                token_type = 'SYM'

            lemma = token_spacy.lemma_.lower()

            lemma = normalize('NFKD', lemma).encode('ASCII', 'ignore').decode('ASCII')

            token = Token(id_token=id_token, text=token_spacy.text, lemma=lemma, pos=token_spacy.pos_,
                          token_type=token_type, is_stopword=token_spacy.is_stop)

            sentence.add_token(token)

        sentence.entities = list(sent_spacy.ents)

        tokens_no_stopwords = [t for t in sentence.tokens if not t.is_stopword and t.token_type == 'WORD']

        sentence.tokens_no_stopwords = tokens_no_stopwords

        document.add_sentence(sentence)

        document.add_tokens_no_stopwords(tokens_no_stopwords)
