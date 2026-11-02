from collections import Counter
import math


class LexicalIndex:
    """Whitespace-tokenized corpus with explicit TF-IDF and BM25 scoring.

    TF-IDF uses raw term counts and smoothed IDF, then cosine similarity.
    BM25 uses a positive Robertson IDF variant and counts each query term once.
    """

    def __init__(self, documents):
        self.documents = [Counter(document.lower().split()) for document in documents]
        if not self.documents:
            raise ValueError("corpus must contain documents")
        self.lengths = [sum(document.values()) for document in self.documents]
        self.average_length = sum(self.lengths) / len(self.documents)
        self.document_frequency = Counter(
            term for document in self.documents for term in document
        )

    def tfidf(self, query):
        def vector(counts):
            return {
                term: count
                * (
                    1
                    + math.log(
                        (1 + len(self.documents)) / (1 + self.document_frequency[term])
                    )
                )
                for term, count in counts.items()
                if term in self.document_frequency
            }

        query_vector = vector(Counter(query.lower().split()))
        query_norm = math.sqrt(sum(value * value for value in query_vector.values()))
        scores = []
        for document in self.documents:
            values = vector(document)
            norm = math.sqrt(sum(value * value for value in values.values()))
            numerator = sum(
                value * values.get(term, 0) for term, value in query_vector.items()
            )
            scores.append(numerator / (norm * query_norm) if norm * query_norm else 0.0)
        return scores

    def bm25(self, query, k1=1.2, b=0.75):
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("invalid BM25 settings")
        scores = []
        for document, length in zip(self.documents, self.lengths):
            score = 0.0
            for term in set(query.lower().split()):
                frequency = document[term]
                if not frequency:
                    continue
                df = self.document_frequency[term]
                idf = math.log1p((len(self.documents) - df + 0.5) / (df + 0.5))
                normalizer = frequency + k1 * (1 - b + b * length / self.average_length)
                score += idf * frequency * (k1 + 1) / normalizer
            scores.append(score)
        return scores
