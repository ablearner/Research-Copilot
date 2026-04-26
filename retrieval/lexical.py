from __future__ import annotations

import math
import re
from collections import Counter


_TOKEN_PATTERN = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)


def tokenize_lexical(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_PATTERN.findall(text or "") if token.strip()]


def bm25_score_texts(
    *,
    query: str,
    texts: list[str],
    k1: float = 1.5,
    b: float = 0.75,
) -> list[float]:
    query_terms = tokenize_lexical(query)
    if not query_terms or not texts:
        return [0.0 for _ in texts]

    tokenized_texts = [tokenize_lexical(text) for text in texts]
    lengths = [len(tokens) for tokens in tokenized_texts]
    avgdl = (sum(lengths) / len(lengths)) if lengths else 0.0
    query_counter = Counter(query_terms)

    document_frequencies: dict[str, int] = {}
    for term in query_counter:
        document_frequencies[term] = sum(1 for tokens in tokenized_texts if term in set(tokens))

    total_docs = len(tokenized_texts)
    scores: list[float] = []
    for tokens, doc_length in zip(tokenized_texts, lengths, strict=True):
        term_frequencies = Counter(tokens)
        score = 0.0
        length_norm = 1.0 - b + b * (doc_length / avgdl) if avgdl > 0 else 1.0
        for term, qf in query_counter.items():
            tf = term_frequencies.get(term, 0)
            if tf <= 0:
                continue
            df = document_frequencies.get(term, 0)
            idf = math.log(1.0 + ((total_docs - df + 0.5) / (df + 0.5)))
            numerator = tf * (k1 + 1.0)
            denominator = tf + k1 * length_norm
            score += qf * idf * (numerator / denominator)
        scores.append(score)
    return scores
