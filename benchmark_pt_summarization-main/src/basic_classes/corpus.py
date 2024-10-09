from src.basic_classes.cluster_documents import ClusterDocuments


class Corpus:

    def __init__(self, name: str, documents: list = None, is_single_doc: bool = True):
        self.name = name
        self.list_cluster_documents: list = []
        self.documents = documents
        self.is_single_doc = is_single_doc

    def add_cluster_documents(self, cluster_documents: ClusterDocuments):
        self.list_cluster_documents.append(cluster_documents)
