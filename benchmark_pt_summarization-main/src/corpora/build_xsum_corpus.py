import os
import corpora_utils as utils


if __name__ == '__main__':

    corpus_name = 'xlsum_pt'

    corpus_path = f'/IFES/Projeto Mestrado/Datasets/Experimentos/pt_summ_benchmark/' \
                  f'corpora/{corpus_name}'

    os.makedirs(corpus_path, exist_ok=True)

    print('\n  Building corpus ...')

    utils.build_xlsum(corpus_path)

    print('\n  Reading corpus ...')

    train_data, valid_data, test_data = utils.read_json_corpus(corpus_path)

    print('\n  Train data:', len(train_data['data']))
    print('    Text:', train_data['data'][0]['text'].replace('\n', ' '))
    print('    Summary:', train_data['data'][0]['target'].replace('\n', ' '))

    print('\n  Valid data:', len(valid_data['data']))
    print('    Text:', valid_data['data'][0]['text'].replace('\n', ' '))
    print('    Summary:', valid_data['data'][0]['target'].replace('\n', ' '))

    print('\n  Test data:', len(test_data['data']))
    print('    Text:', test_data['data'][0]['text'].replace('\n', ' '))
    print('    Summary:', test_data['data'][0]['target'].replace('\n', ' '))
