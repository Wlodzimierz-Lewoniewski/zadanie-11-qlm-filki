import re
import string
from typing import List, Dict
from collections import Counter
import math
from math import log


class Document:
    def __init__(self, index: int, content: str):
        self.index = index
        self.content = content
        self.tokens = self._tokenize(content)
        self.term_frequencies = Counter(self.tokens)
        self.doc_length = len(self.tokens)
    
    def _tokenize(self, text: str) -> List[str]:
        table = str.maketrans('', '', string.punctuation)
        return text.translate(table).lower().split()


class CorpusProcessor:
    def __init__(self, documents: List[Document], lambda_param: float = 0.5):
        self.documents = documents
        self.lambda_param = lambda_param
        self.collection_term_frequencies = self._calculate_collection_frequencies()
        self.collection_length = sum(self.collection_term_frequencies.values())
        
    def _calculate_collection_frequencies(self) -> Counter:
        collection_freqs = Counter()
        for doc in self.documents:
            collection_freqs.update(doc.term_frequencies)
        return collection_freqs
    
    def calculate_query_likelihood(self, query_tokens: List[str], document: Document) -> float:
        log_probability = 0.0
        
        for token in query_tokens:
            p_t_md = document.term_frequencies.get(token, 0) / document.doc_length if document.doc_length > 0 else 0
            p_t_mc = self.collection_term_frequencies.get(token, 0) / self.collection_length if self.collection_length > 0 else 0
            p_t_d = self.lambda_param * p_t_md + (1 - self.lambda_param) * p_t_mc
            p_t_d = max(p_t_d, 1e-10)
            log_probability += log(p_t_d)
        
        return log_probability


class QueryLikelihoodModel:
    def __init__(self, lambda_param: float = 0.5):
        self.lambda_param = lambda_param
        
    def process_input(self, n: int, document_contents: List[str], query: str) -> List[int]:
        documents = [Document(i, content) for i, content in enumerate(document_contents)]
        corpus_processor = CorpusProcessor(documents, self.lambda_param)
        query_doc = Document(0, query)
        query_tokens = query_doc.tokens
        
        doc_probabilities = []
        for doc in documents:
            probability = corpus_processor.calculate_query_likelihood(query_tokens, doc)
            doc_probabilities.append((doc.index, probability))
        
        doc_probabilities.sort(key=lambda x: (-x[1], x[0]))
        return [idx for idx, _ in doc_probabilities]


def main():
    n = int(input().strip())
    documents = []
    for _ in range(n):
        documents.append(input().strip())
    query = input().strip()
    qlm = QueryLikelihoodModel()
    result = qlm.process_input(n, documents, query)
    print(result)


if __name__ == "__main__":
    main()
