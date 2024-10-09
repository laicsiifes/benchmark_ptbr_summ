from corpora.corpora_utils import read_cstnews_corpus, read_xsum_corpus


def test_read_cstnews():
    cstnews_path = 'D:\\Hilario\\Pesquisa\\Recursos\\Sumarizacao\\Corpora\\CSTNews 6.0'
    corpus = read_cstnews_corpus(cstnews_path)
    print(f'\n  Total of clusters: {len(corpus.list_cluster_documents)}')


def test_read_xsum():
    xsum_path = 'D:\\Hilario\\Pesquisa\\Recursos\\Sumarizacao\\Corpora\\xlsum_pt'
    corpus = read_xsum_corpus(xsum_path)
    print('\n  Train:', len(corpus['train'].documents))
    print('  Valid:', len(corpus['valid'].documents))
    print('  Test:', len(corpus['test'].documents))
    print('\n  Test document')
    for doc in corpus['test'].documents[:10]:
        print('\n    Name:', doc.name)
        print('      Text:', doc.text.replace('\n', ' '))
        print('      Reference Summary:', doc.reference_summary.replace('\n', ' '))


if __name__ == '__main__':

    print('\nCorpus CST News\n')

    test_read_cstnews()

    print('\n\nCorpus XSUM PT')

    test_read_xsum()
