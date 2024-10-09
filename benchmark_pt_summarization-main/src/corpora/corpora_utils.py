import os
import re
import json
import numpy as np
import spacy
from tqdm import tqdm
import sys
import csv

from datasets import load_dataset
from src.basic_classes.corpus import Corpus
from src.basic_classes.cluster_documents import ClusterDocuments
from src.basic_classes.document import Document
from src.basic_classes.summary import Summary
from datasets import Dataset
import pandas as pd


def build_xlsum(corpus_path: str):
    dataset = load_dataset('GEM/xlsum', 'portuguese')
    train_data = {'data': []}
    valid_data = {'data': []}
    test_data = {'data': []}
    for i, train in enumerate(dataset['train']):
        train['id'] = i + 1
        train_data['data'].append(train)
    for j, val in enumerate(dataset['validation']):
        val['id'] = j + 1
        valid_data['data'].append(val)
    for k, test in enumerate(dataset['test']):
        test['id'] = k + 1
        test_data['data'].append(test)
    train_file = os.path.join(corpus_path, 'train.json')
    valid_file = os.path.join(corpus_path, 'valid.json')
    test_file = os.path.join(corpus_path, 'test.json')
    with open(train_file, 'w') as file:
        json.dump(train_data, file, indent=4)
    with open(valid_file, 'w') as file:
        json.dump(valid_data, file, indent=4)
    with open(test_file, 'w') as file:
        json.dump(test_data, file, indent=4)


def read_json_corpus(corpus_path):
    train_file = os.path.join(corpus_path, 'train.json')
    valid_file = os.path.join(corpus_path, 'valid.json')
    test_file = os.path.join(corpus_path, 'test.json')
    with open(train_file, 'r') as file:
        train_data = json.load(file)
    with open(valid_file, 'r') as file:
        valid_data = json.load(file)
    with open(test_file, 'r') as file:
        test_data = json.load(file)
    return train_data, valid_data, test_data


"""
    Download: https://sites.icmc.usp.br/taspardo/sucinto/cstnews.html
"""


def read_cstnews_corpus(corpus_path: str) -> Corpus:
    assert os.path.exists(corpus_path), f'ERROR. DIRECTORY {corpus_path} NOT FOUND'

    list_cluster_names = os.listdir(corpus_path)

    corpus = Corpus(name='cst_news', is_single_doc=False)

    for cluster_dir_name in list_cluster_names:

        if cluster_dir_name[0] == 'C':

            cluster_name = cluster_dir_name.split('_')[0]

            cluster_documents = ClusterDocuments(cluster_name)

            texts_dir = os.path.join(corpus_path, cluster_dir_name, 'Textos-fonte')
            extractive_summaries_dir = os.path.join(corpus_path, cluster_dir_name, 'Sumarios', 'Novos sumarios',
                                                    'Extratos')
            abstractive_summaries_dir = os.path.join(corpus_path, cluster_dir_name, 'Sumarios', 'Novos sumarios',
                                                     'Abstracts')

            for extractive_summary_name in os.listdir(extractive_summaries_dir):
                extractive_summary_path = os.path.join(extractive_summaries_dir, extractive_summary_name)
                try:
                    with open(file=extractive_summary_path, mode='r', encoding='utf-8') as file:
                        reference_summary = file.read()
                    reference_summary = re.sub('<.*?>', '', reference_summary)
                    cluster_documents.add_extractive_reference_summary(reference_summary)
                except UnicodeDecodeError:
                    pass

            for abstractive_summary_name in os.listdir(abstractive_summaries_dir):
                abstractive_summary_path = os.path.join(abstractive_summaries_dir, abstractive_summary_name)
                try:
                    with open(file=abstractive_summary_path, mode='r', encoding='utf-8') as file:
                        reference_summary = file.read()
                    reference_summary = re.sub('<.*?>', '', reference_summary)
                    cluster_documents.add_abstractive_reference_summary(reference_summary)
                except UnicodeDecodeError:
                    pass

            id_document = 1

            for document_file_name in os.listdir(texts_dir):

                if '_titulo' not in document_file_name:

                    document_file_name_fragments = document_file_name.split('_')
                    document_name = document_file_name_fragments[0] + '_' + document_file_name_fragments[1]

                    title_file_name = document_file_name.replace('.txt', '_titulo.txt')

                    text_path = os.path.join(texts_dir, document_file_name)
                    title_path = os.path.join(texts_dir, title_file_name)

                    with open(file=text_path, mode='r', encoding='utf-8') as file:
                        text = file.read()

                    title = None

                    if os.path.exists(title_path):
                        with open(file=title_path, mode='r', encoding='utf-8') as file:
                            title = file.read()

                    text = text.replace('\n\n', '\n')
                    text = text.replace('', '')
                    text = text.replace('', '')
                    text = text.replace('', '')
                    text = text.replace('', '')

                    document = Document(id_document=id_document, name=document_name, title=title,
                                        text=text)

                    cluster_documents.add_document(document)

                    id_document += 1

            corpus.add_cluster_documents(cluster_documents)

    return corpus


"""
    Download: http://www.nilc.icmc.usp.br/nilc/tools/TeMario.zip
"""


def read_temario_corpus(corpus_path: str) -> Corpus:
    articles_dir = os.path.join(corpus_path, 'Textos-fonte', 'Textos-fonte com título')
    summaries_dir = os.path.join(corpus_path, 'Sumários', 'Sumários manuais')

    articles_files_names = os.listdir(articles_dir)

    documents = []

    for id_document, article_file_name in enumerate(articles_files_names, start=1):

        article_file_path = os.path.join(articles_dir, article_file_name)

        with open(file=article_file_path, mode='r', encoding='latin-1') as file:
            lines = file.readlines()

        summary_file_path = os.path.join(summaries_dir, f'Sum-{article_file_name}')

        with open(file=summary_file_path, mode='r', encoding='latin-1') as file:
            summary = file.read()

        title = lines[0]

        text = ''

        for line in lines[1:]:
            if len(line.strip()) > 0:
                text += f'{line} '

        text = text.strip()

        document_name = article_file_name.replace('.txt', '')

        document = Document(id_document=id_document, name=document_name, title=title,
                            text=text, reference_summary=summary)

        documents.append(document)

    corpus = Corpus(name='temario', documents=documents, is_single_doc=True)

    return corpus


def filter_dataset_empty_text(dataset):
    df_pandas = pd.DataFrame(dataset)
    df_pandas['text_length'] = df_pandas['text'].apply(lambda x: len(x.split()))
    df_pandas['summary_length'] = df_pandas['summary'].apply(lambda x: len(x.split()))
    df_pandas = df_pandas[df_pandas['text_length'] >= 150].reset_index(drop=True)
    df_pandas = df_pandas[df_pandas['summary_length'] >= 10].reset_index(drop=True)
    filtered_dataset = Dataset.from_pandas(df_pandas.drop(columns=['text_length']))
    return filtered_dataset


def filter_dataset_empty_text_old(dataset):
    list_index = []
    for i in range(len(dataset)):
        # print('STRING VAZIA', i)
        if dataset[i]['text']:
            text_dataset = dataset[i]['text'].split()
            if len(text_dataset) < 10:
                list_index.append(i)
        else:
            list_index.append(i)
    df_pandas = pd.DataFrame(dataset)
    for i in range(len(list_index)):
        df_pandas = df_pandas.drop(list_index[i])
    filtered_dataset = Dataset.from_pandas(df_pandas)
    return filtered_dataset


def read_recognasumm_dataset(dataset):

    train_dataset = dataset['train']
    val_dataset = dataset['validation']
    test_dataset = dataset['test']

    train_dataset = train_dataset.rename_column("index", "id")
    train_dataset = train_dataset.rename_column("Titulo", "title")
    train_dataset = train_dataset.rename_column("Noticia", "text")
    train_dataset = train_dataset.rename_column("Sumario", "summary")

    val_dataset = val_dataset.rename_column("index", "id")
    val_dataset = val_dataset.rename_column("Titulo", "title")
    val_dataset = val_dataset.rename_column("Noticia", "text")
    val_dataset = val_dataset.rename_column("Sumario", "summary")

    test_dataset = test_dataset.rename_column("index", "id")
    test_dataset = test_dataset.rename_column("Titulo", "title")
    test_dataset = test_dataset.rename_column("Noticia", "text")
    test_dataset = test_dataset.rename_column("Sumario", "summary")

    val_dataset_filtered = filter_dataset_empty_text(val_dataset)
    train_dataset_filtered = filter_dataset_empty_text(train_dataset)
    test_dataset_filtered = filter_dataset_empty_text(test_dataset)

    train_docs = build_documents_dataset(train_dataset_filtered)
    val_docs = build_documents_dataset(val_dataset_filtered)
    test_docs = build_documents_dataset(test_dataset_filtered)

    corpus = {
        'train': Corpus(name='train', documents=train_docs, is_single_doc=True),
        'valid': Corpus(name='val', documents=val_docs, is_single_doc=True),
        'test': Corpus(name='test', documents=test_docs, is_single_doc=True)
    }

    return corpus


def read_xlsum_dataset(dataset):
    train_dataset = dataset['train']
    val_dataset = dataset['validation']
    test_dataset = dataset['test']

    train_dataset_filtered = filter_dataset_empty_text(train_dataset)
    val_dataset_filtered = filter_dataset_empty_text(val_dataset)
    test_dataset_filtered = filter_dataset_empty_text(test_dataset)

    train_docs = build_documents_dataset(train_dataset_filtered)
    val_docs = build_documents_dataset(val_dataset_filtered)
    test_docs = build_documents_dataset(test_dataset_filtered)

    corpus = {
        'train': Corpus(name='train', documents=train_docs, is_single_doc=True),
        'valid': Corpus(name='val', documents=val_docs, is_single_doc=True),
        'test': Corpus(name='test', documents=test_docs, is_single_doc=True)
    }

    return corpus


def read_xsum_corpus(corpus_path: str) -> dict:
    train_file = os.path.join(corpus_path, 'train.json')
    valid_file = os.path.join(corpus_path, 'valid.json')
    test_file = os.path.join(corpus_path, 'test.json')

    with open(train_file, 'r') as file:
        train_data = json.load(file)

    with open(valid_file, 'r') as file:
        valid_data = json.load(file)

    with open(test_file, 'r') as file:
        test_data = json.load(file)

    train_docs = build_documents(train_data)
    val_docs = build_documents(valid_data)
    test_docs = build_documents(test_data)

    corpus = {
        'train': Corpus(name='train', documents=train_docs, is_single_doc=True),
        'valid': Corpus(name='valid', documents=val_docs, is_single_doc=True),
        'test': Corpus(name='test', documents=test_docs, is_single_doc=True)
    }

    return corpus


def build_documents_dataset(dataset_):
    documents = []
    for data in dataset_:
        document = Document()
        document.name = str(data['id'])
        document.title = data['title']
        text = data['text'].replace('\n', ' ')
        text = re.sub(r'\s\s+', ' ', text)
        document.text = text
        document.reference_summary = data['summary']
        documents.append(document)
    return documents


def build_documents(data_: dict) -> list:
    documents = []
    for data in data_['data']:
        document = Document()
        document.name = str(data['id'])
        document.title = data['title']
        text = data['text'].replace('\n', ' ')
        text = re.sub(r'\s\s+', ' ', text)
        document.text = text
        document.reference_summary = data['references'][0]
        documents.append(document)
    return documents


def save_summary(directory_name: str, summary: Summary, summaries_dir: str):
    document_dir = os.path.join(summaries_dir, directory_name.lower())
    os.makedirs(document_dir, exist_ok=True)
    summary_path = os.path.join(document_dir, summary.name + '.txt')
    with open(summary_path, 'w', encoding='utf-8') as file:
        file.write(summary.text)


def save_summaries(corpus: Corpus, summaries_dir: str):
    for document in corpus.documents:
        document_dir = os.path.join(summaries_dir, document.name.lower())
        os.makedirs(document_dir, exist_ok=True)
        for summary_name, summary in document.candidate_summaries:
            summary_path = os.path.join(document_dir, summary_name + '.txt')
            with open(summary_path, 'w', encoding='utf-8') as file:
                file.write(document.summary)


def read_summaries_json(corpus: Corpus, summaries_dir: str):
    systems_files_names = os.listdir(summaries_dir)
    systems_files_names = [file_name for file_name in systems_files_names if file_name.endswith('.json')]
    doc_dict = {}
    for doc in corpus.documents:
        doc_dict[doc.name] = doc
    for file_name in systems_files_names:
        summaries_path = os.path.join(summaries_dir, file_name)
        with open(summaries_path, encoding='utf-8') as file:
            list_systems_summaries = json.load(file)
        summary_name = file_name.replace('.json', '')
        for system_summaries in list_systems_summaries:
            doc_name = str(system_summaries['name'])
            if doc_name in doc_dict:
                doc = doc_dict[doc_name]
                doc.add_candidate_summary(Summary(name=summary_name.replace('.txt', ''),
                                                  text=system_summaries['generated_summary']))


def read_summaries(corpus: Corpus, summaries_dir: str) -> None:
    for doc in corpus.documents:
        doc_dir = os.path.join(summaries_dir, doc.name)
        summaries_names = os.listdir(doc_dir)
        for summary_name in summaries_names:
            summary_path = os.path.join(doc_dir, summary_name)
            with open(summary_path, encoding='utf-8') as file:
                text_summary = file.read()
            doc.add_candidate_summary(Summary(name=summary_name.replace('.txt', ''), text=text_summary))

def count_metrics(texto, nlp):

    doc = nlp(texto)

    len_word = len([token for token in doc if not token.is_punct and not token.is_space])
    len_sentence = len(list(doc.sents))
    
    return len_word, len_sentence

def count_word(texto, nlp):
    doc = nlp(texto)
    return len([token for token in doc])
    

def metrics_dataset_recognasumm(dataset_rec, metrics_path, corpus_name):
    train_dataset = dataset_rec['train']
    val_dataset = dataset_rec['validation']
    test_dataset = dataset_rec['test']

    train_dataset = train_dataset.rename_column("Noticia", "text")
    train_dataset = train_dataset.rename_column("Sumario", "summary")

    val_dataset = val_dataset.rename_column("Noticia", "text")
    val_dataset = val_dataset.rename_column("Sumario", "summary")

    test_dataset = test_dataset.rename_column("Noticia", "text")
    test_dataset = test_dataset.rename_column("Sumario", "summary")

    train_dataset_filtered = filter_dataset_empty_text(train_dataset)
    val_dataset_filtered = filter_dataset_empty_text(val_dataset)
    test_dataset_filtered = filter_dataset_empty_text(test_dataset)

    nlp = spacy.load('pt_core_news_lg')

    datasets = {
        'train_dataset': train_dataset_filtered,
        'val_dataset': val_dataset_filtered,
        'test_dataset': test_dataset_filtered
    }

    for dataset_name, dataset in datasets.items():
        qtdFrases = []
        qtdPalavras = []
        
        with tqdm(total=len(dataset), file=sys.stdout, colour='red', desc=f'Evaluating Metrics from - {dataset_name}') as pbar:
            for item in dataset:

                total_words, num_sents = count_metrics(item['text'], nlp)

                qtdFrases.append(num_sents)
                qtdPalavras.append(total_words)

                pbar.update(1)

        dsvPadraoFrases = np.std(qtdFrases)
        dsvPadraoPalavras = np.std(qtdPalavras)
        mediaFrases = np.average(qtdFrases)
        mediaPalavras = np.average(qtdPalavras)

        csv_path = os.path.join(metrics_path, corpus_name + dataset_name + '.csv')
        
        with open(csv_path, 'w', encoding='utf-8', newline='') as csvfile:
           
            writer = csv.writer(csvfile, delimiter=',')

            writer.writerow(['Metrica', 'Valor'])
            writer.writerow(['Quantidade de Frases', sum(qtdFrases)])
            writer.writerow(['Media de Frases por documento', mediaFrases])
            writer.writerow(['Desvio Padrao Frases', dsvPadraoFrases])
            writer.writerow(['Quantidade de Palavras', sum(qtdPalavras)])
            writer.writerow(['Media de Palavras por documento', mediaPalavras])
            writer.writerow(['Desvio Padrao Palavras', dsvPadraoPalavras])


def metrics_dataset_temario(corpus, metrics_path, corpus_name):
        
    nlp = spacy.load('pt_core_news_lg')

    qtdFrases = []
    qtdPalavras = []

    for document in corpus.documents:

        total_words, num_sents = count_metrics(document.text, nlp)

        qtdFrases.append(num_sents)
        qtdPalavras.append(total_words)

    dsvPadraoFrases = np.std(qtdFrases)
    dsvPadraoPalavras = np.std(qtdPalavras)
    mediaFrases = np.average(qtdFrases)
    mediaPalavras = np.average(qtdPalavras)
  
    csv_path = os.path.join(metrics_path, corpus_name + '.csv')
    
    with open(csv_path, 'w', encoding='utf-8', newline='') as csvfile:
        
        writer = csv.writer(csvfile, delimiter=',')

        writer.writerow(['Metrica', 'Valor'])
        writer.writerow(['Quantidade de Frases', sum(qtdFrases)])
        writer.writerow(['Media de Frases por documento', mediaFrases])
        writer.writerow(['Desvio Padrao Frases', dsvPadraoFrases])
        writer.writerow(['Quantidade de Palavras', sum(qtdPalavras)])
        writer.writerow(['Media de Palavras por documento', mediaPalavras])
        writer.writerow(['Desvio Padrao Palavras', dsvPadraoPalavras])



def metrics_dataset_cstnews(corpus, metrics_path, corpus_name):
        
    nlp = spacy.load('pt_core_news_lg')

    qtdFrases = []
    qtdPalavras = []
    
    for cluster_documents in corpus.list_cluster_documents:
        
        full_text = ' '.join([doc.text for doc in cluster_documents.documents])

        total_words, num_sents = count_metrics(full_text, nlp)

        qtdFrases.append(num_sents)
        qtdPalavras.append(total_words)

    dsvPadraoFrases = np.std(qtdFrases)
    dsvPadraoPalavras = np.std(qtdPalavras)
    mediaFrases = np.average(qtdFrases)
    mediaPalavras = np.average(qtdPalavras)

    csv_path = os.path.join(metrics_path, corpus_name + '.csv')
    
    with open(csv_path, 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Metrica', 'Valor'])
        writer.writerow(['Quantidade de Frases', sum(qtdFrases)])
        writer.writerow(['Media de Frases por documento', mediaFrases])
        writer.writerow(['Desvio Padrao Frases', dsvPadraoFrases])
        writer.writerow(['Quantidade de Palavras', sum(qtdPalavras)])
        writer.writerow(['Media de Palavras por documento', mediaPalavras])
        writer.writerow(['Desvio Padrao Palavras', dsvPadraoPalavras])