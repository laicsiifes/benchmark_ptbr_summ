from src.basic_classes.document import Document
from bert_score import BERTScorer


"""
    https://github.com/Tiiiger/bert_score
    
    Foi necessário inserir dentro da biblioteca o caminho para o bertimbau:
     
        "pt": "neuralmind/bert-large-portuguese-cased"
"""


def evaluate_summaries_bert_score(document: Document, bert_scorer: BERTScorer):
    for summary in document.candidate_summaries:
        candidates = [summary.text]
        if document.references_summaries is None:
            reference_summaries = [document.reference_summary]
        else:
            reference_summaries = document.references_summaries
        reference_summaries = [[r.replace('\n', ' ').lower()] for r in reference_summaries]
        precision, recall, f_measure = bert_scorer.score(candidates, reference_summaries)
        bert_score_measures = {
            'bert_score': {
                'p': precision.numpy()[0],
                'r': recall.numpy()[0],
                'f': f_measure.numpy()[0]
            }
        }
        summary.rouge_scores = bert_score_measures
