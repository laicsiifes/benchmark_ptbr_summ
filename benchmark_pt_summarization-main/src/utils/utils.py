from src.basic_classes.sentence import Sentence


def compute_sentence_similarity(sentence_1: Sentence, sentence_2: Sentence) -> float:
    lemmas_1 = set([t.lemma for t in sentence_1.tokens_no_stopwords])
    lemmas_2 = set([t.lemma for t in sentence_2.tokens_no_stopwords])
    intersection = len(lemmas_1.intersection(lemmas_2))
    union = len(lemmas_1.union(lemmas_2))
    jaccard_similarity = intersection / union
    return jaccard_similarity
