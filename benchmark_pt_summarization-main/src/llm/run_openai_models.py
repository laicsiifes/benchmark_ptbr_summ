import os
# import openai

from dotenv import load_dotenv
from src.corpora.corpora_utils import read_cstnews_corpus, read_temario_corpus
from src.summarizers.openai_summarizers import (estimate_cost_single, estimate_cost_multi,
                                                summarize_multidocument, summarize_single_document)


if __name__ == '__main__':

    load_dotenv()

    # corpus_name = 'temario'
    corpus_name = 'cstnews'

    # model_engine = 'text-davinci-003'
    # model_engine = 'gpt-3.5-turbo-16k'
    # model_engine = 'gpt-4o-mini'
    model_engine = 'gpt-4o'

    temperature = 0.3

    os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

    summary_length = 150

    if model_engine == 'text-davinci-003':
        model_name = f'davinci_003_{temperature}'.replace('.', '')
    elif model_engine == 'text-davinci-003':
        model_name = f'gpt_35_turbo_{temperature}'.replace('.', '')
    elif model_engine == 'gpt-4o-mini':
        model_name = f'gpt_4o_mini_{temperature}'.replace('.', '')
    elif model_engine == 'gpt-4o':
        model_name = f'gpt_4o_{temperature}'.replace('.', '')
    else:
        print('Error!')
        exit(-1)

    corpus_path = f'/mnt/Novo Volume/Hilario/Pesquisa/Experimentos/pt_summ_benchmark/' \
                  f'corpora/{corpus_name}'

    summaries_dir = (f'/mnt/Novo Volume/Hilario/Pesquisa/Experimentos/pt_summ_benchmark/'
                     f'summaries_openai/{corpus_name}')

    os.makedirs(summaries_dir, exist_ok=True)

    print(f'\nCorpus: {corpus_name}')

    corpus = None
    summarize = None
    model_path = None
    estimate_cost = None

    if corpus_name == 'cstnews':
        corpus = read_cstnews_corpus(corpus_path)
        summarize = summarize_multidocument
        estimate_cost = estimate_cost_multi
    elif corpus_name == 'temario':
        corpus = read_temario_corpus(corpus_path)
        summarize = summarize_single_document
        estimate_cost = estimate_cost_single
    else:
        print(f'Error. corpus_name must be "cstnews" or "temario".')
        exit(-1)

    print(f'\nSummarization Model: {model_name}\n')

    estimate_cost(corpus, summary_length, model_engine)

    print('\n')

    summarize(corpus, summary_length, temperature, model_engine, model_name, summaries_dir)
