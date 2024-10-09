import numpy as np
import os

from src.basic_classes.document import Document
from src.basic_classes.corpus import Corpus
from rouge import Rouge

"""
    https://github.com/Diego999/py-rouge
"""


def evaluate_summaries(document: Document, is_use_stemming: bool, limit_words: int = 100):
    evaluator = Rouge(metrics=['rouge-n', 'rouge-l'], max_n=2, limit_length=True, length_limit=limit_words,
                      length_limit_type='words', apply_avg=True, apply_best=False, alpha=0.5,
                      weight_factor=1.2, stemming=is_use_stemming)
    for summary in document.candidate_summaries:
        if document.references_summaries is None:
            references_summaries = [document.reference_summary.lower()]
        else:
            references_summaries = document.references_summaries
        rouge_scores = evaluator.get_scores(summary.text.replace('\n', ' ').lower(), references_summaries)
        summary.rouge_scores = rouge_scores


def generate_document_report(document: Document, directory: str):
    avg_scores_configurations = {}
    for summary in document.candidate_summaries:
        rouge_variations = {}
        for variation, metrics in summary.rouge_scores.items():
            rouge_variations[variation] = {'r': metrics['r'], 'p': metrics['p'], 'f': metrics['f']}
        avg_scores_configurations[summary.name] = rouge_variations
    save_report_document(avg_scores_configurations, directory)


def save_report_document(avg_scores_configurations: dict, directory: str):
    report_results = {}
    for configuration, rouge_variations in avg_scores_configurations.items():
        for rouge_variation, metrics_values in rouge_variations.items():
            if rouge_variation in report_results:
                configurations_results = report_results[rouge_variation]
            else:
                configurations_results = {}
            configurations_results[configuration] = metrics_values
            report_results[rouge_variation] = configurations_results
    for rouge_variation, configurations_results in report_results.items():
        results_report_file = os.path.join(directory, rouge_variation + '.csv')
        report = 'System;Recall;Precision;F-measure\n'
        for configuration, results in configurations_results.items():
            report += f'{configuration};{str(results["r"])};{str(results["p"])};{str(results["f"])}\n'
        report = report[:-1]
        with open(results_report_file, 'w') as file:
            file.write(report)


def generate_report(corpus: Corpus, summaries_dir: str):
    avg_scores_configurations = {}
    for document in corpus.documents:
        for summary in document.candidate_summaries:
            rouge_variations = {}
            if summary.name in avg_scores_configurations:
                rouge_variations = avg_scores_configurations[summary.name]
            for variation, metrics in summary.rouge_scores.items():
                if variation in rouge_variations:
                    metrics_values = rouge_variations[variation]
                else:
                    metrics_values = {'r': [], 'p': [], 'f': []}
                metrics_values['r'].append(metrics['r'])
                metrics_values['p'].append(metrics['p'])
                metrics_values['f'].append(metrics['f'])
                rouge_variations[variation] = metrics_values
            avg_scores_configurations[summary.name] = rouge_variations
    save_report(avg_scores_configurations, summaries_dir)


def save_report(avg_scores_configurations: dict, summaries_dir: str):
    report_results = {}
    for configuration, rouge_variations in avg_scores_configurations.items():
        for rouge_variation, metrics_values in rouge_variations.items():
            if rouge_variation in report_results:
                configurations_results = report_results[rouge_variation]
            else:
                configurations_results = {}
            configurations_results[configuration] = metrics_values
            report_results[rouge_variation] = configurations_results
    for rouge_variation, configurations_results in report_results.items():
        results_report_file = os.path.join(summaries_dir, f'{rouge_variation}.csv')
        report = 'System;Recall;Standard Deviation;Precision;Standard Deviation;F-measure;Standard Deviation\n'
        for configuration, results in configurations_results.items():
            report += f'{configuration};'
            mean_recall = np.mean(results['r'])
            mean_precision = np.mean(results['p'])
            mean_f_score = np.mean(results['f'])
            if len(results['r']) > 1:
                report += f'{str(mean_recall)};{str(np.std(results["r"]))};{str(mean_precision)};' \
                          f'{str(np.std(results["p"]))};{str(mean_f_score)};{str(np.std(results["f"]))}\n'
            else:
                report += f'{str(mean_recall)};0;{str(mean_precision)};0;{str(mean_f_score)};0\n'
        report = report[:-1]
        report = report.replace('.', ',')
        with open(results_report_file, 'w') as file:
            file.write(report)
