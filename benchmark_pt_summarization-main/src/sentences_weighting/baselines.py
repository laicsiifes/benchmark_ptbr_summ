from enum import Enum


class BaselinesEnum(Enum):

    WORD_FREQUENCY = 'word_freq'
    TF_ISF = 'tf_isf'
    SENTENCE_POSITION = 'sentences_position'
    SENTENCE_CENTRALITY = 'sentence_centrality'
    BUSHY_PATH = 'bushy_path'
    AGGREGATE_SIMILARITY = 'aggregate_similarity'
    NAMED_ENTITIES_FREQUENCY = 'named_entities_frequency'
    TEXTRANK = 'textrank'

    def __repr__(self):
        return self.value
