"""Лексический поиск BM25 по фрагментам стенограмм, без внешних API."""

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
import re

from knowledge import Chunk


def tokenize(text: str) -> list[str]:
    """Выделить слова и числа, привести регистр и заменить ё на е.

    Без стемминга: разные формы слова пока считаются разными словами.
    """
    return re.findall(r"[а-яa-z0-9]+", text.lower().replace("ё", "е"))


@dataclass(frozen=True)
class SearchResult:
    """Найденный фрагмент и score поиска (не вероятность правильности)."""

    chunk: Chunk
    score: float


class LexicalSearch:
    """Индекс BM25: редкие совпадения важнее, длина текста учитывается.

    Индекс строится один раз. При запросе просматриваются только фрагменты,
    содержащие хотя бы одно слово запроса. При равных score сохраняется
    исходный порядок фрагментов. Параметры BM25 фиксированы: k1=1.5, b=0.75.
    """

    def __init__(self, chunks: list[Chunk]) -> None:
        """Построить частоты слов, обратный индекс и нормировку длины."""
        self._chunks = tuple(chunks)
        if len({chunk.chunk_id for chunk in chunks}) != len(chunks):
            raise ValueError("Идентификаторы фрагментов должны быть уникальны")
        self._postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        lengths = []
        for index, chunk in enumerate(chunks):
            counts = Counter(tokenize(chunk.text))
            lengths.append(sum(counts.values()))
            for term, count in counts.items():
                self._postings[term].append((index, count))
        average_length = sum(lengths) / len(lengths) if lengths else 0.0
        self._length_norm = [
            1.5 * (1 - 0.75 + 0.75 * length / average_length)
            if average_length else 1.5
            for length in lengths
        ]
        self._idf = {
            term: math.log(1 + (len(chunks) - len(postings) + 0.5) / (len(postings) + 0.5))
            for term, postings in self._postings.items()
        }

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Вернуть до top_k совпадений; пустой/неизвестный запрос даёт []."""
        if top_k < 1:
            raise ValueError("top_k должен быть положительным")
        scores: dict[int, float] = defaultdict(float)
        for term in sorted(set(tokenize(query))):
            for index, count in self._postings.get(term, ()):
                scores[index] += (
                    self._idf[term] * count * 2.5 / (count + self._length_norm[index])
                )
        ranked = sorted(scores, key=lambda index: (-scores[index], index))[:top_k]
        return [SearchResult(self._chunks[index], scores[index]) for index in ranked]
