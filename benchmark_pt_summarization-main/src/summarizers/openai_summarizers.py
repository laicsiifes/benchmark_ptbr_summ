import sys
import os

from src.basic_classes.corpus import Corpus
from tqdm import tqdm
from tokencost import calculate_prompt_cost
from openai import OpenAI
from src.basic_classes.summary import Summary
from src.corpora.corpora_utils import save_summary


def estimate_cost_single(corpus: Corpus, summary_length: int, model_name: str):

    total_cost = 0

    with tqdm(total=len(corpus.documents), file=sys.stdout, colour='yellow', desc='Estimating') as pbar:

        for document in corpus.documents:

            full_text = document.text.replace('\n', ' ')

            prompt = f"""Escreva um resumo em PORTUGUÊS DO BRASIL para o artigo de notícias a seguir com no MÁXIMO {summary_length} palavras.\nARTIGO: ```{full_text}```."""

            prompt *= 2

            total_cost += calculate_prompt_cost(prompt=prompt, model=model_name)

            pbar.update(1)

    print(f'\nEstimated Cost: ${total_cost}')


def estimate_cost_multi(corpus: Corpus, summary_length: int, model_name: str):

    total_cost = 0

    with tqdm(total=len(corpus.list_cluster_documents), file=sys.stdout, colour='yellow',
              desc='Estimating') as pbar:

        for cluster_documents in corpus.list_cluster_documents:

            full_text = ' '.join([doc.text for doc in cluster_documents.documents])

            full_text = full_text.replace('\n', ' ')

            prompt = f"""Escreva um resumo em PORTUGUÊS DO BRASIL para o artigo de notícias a seguir com no MÁXIMO {summary_length} palavras.\nARTIGO: ```{full_text}```."""

            prompt *= 2

            total_cost += calculate_prompt_cost(prompt=prompt, model=model_name)

            pbar.update(1)

    print(f'\nEstimated Cost: ${total_cost}')


def summarize_single_document(corpus: Corpus, summary_length: int, temperature: float,
                              model_engine: str, model_name: str, summaries_dir: str):

    client = OpenAI()

    with tqdm(total=len(corpus.documents), file=sys.stdout, colour='red', desc='Summarizing') as pbar:

        for document in corpus.documents:

            summary_path = os.path.join(summaries_dir, document.name.lower(), f'{model_name}.txt')

            if os.path.exists(summary_path):
                pbar.update(1)
                continue

            full_text = document.text.replace('\n', ' ')

            # prompt_base = f"""Escreva um resumo em PORTUGUÊS DO BRASIL para o artigo de notícias a seguir com no MÁXIMO {summary_length} palavras. ARTIGO: ```{full_text}```."""

            prompt_base = f'Escreva um resumo com {summary_length} palavras para o seguinte artigo:\n{full_text}\n'

            response = client.chat.completions.create(
                model=model_engine,
                temperature=temperature, max_tokens=256, top_p=1,
                frequency_penalty=0, presence_penalty=0,
                messages=[
                    {'role': 'user', 'content': prompt_base},
                ],
            )

            summary_text = response.choices[0].message.content

            summary_text = summary_text.replace('\n\n', '')

            summary = Summary(name=model_name, text=summary_text)

            save_summary(document.name, summary, summaries_dir)

            pbar.update(1)


def summarize_multidocument(corpus: Corpus, summary_length: int, temperature: float,
                            model_engine: str, model_name: str, summaries_dir: str):

    client = OpenAI()

    with tqdm(total=len(corpus.list_cluster_documents), file=sys.stdout, colour='red',
              desc='Summarizing') as pbar:

        for cluster_documents in corpus.list_cluster_documents:

            summary_path = os.path.join(summaries_dir, cluster_documents.name.lower(),
                                        f'{model_name}.txt')

            if os.path.exists(summary_path):
                pbar.update(1)
                continue

            full_text = ' '.join([doc.text for doc in cluster_documents.documents])

            full_text = full_text.replace('\n', ' ')

            prompt_base = f'Escreva um resumo com {summary_length} palavras para o seguinte artigo:\n{full_text}\n'

            response = client.chat.completions.create(
                model=model_engine,
                temperature=temperature, max_tokens=256, top_p=1,
                frequency_penalty=0, presence_penalty=0,
                messages=[
                    {'role': 'user', 'content': prompt_base},
                ]
            )

            summary_text = response.choices[0].message.content

            summary_text = summary_text.replace('\n\n', ' ').strip()

            summary = Summary(name=model_name, text=summary_text)

            save_summary(cluster_documents.name, summary, summaries_dir)

            pbar.update(1)
