"""Семантический поиск по косинусному сходству Giga Embeddings / Cloud.ru."""

from functools import cache
from tokenizers import Tokenizer
import numpy as np

from embeddings import EmbeddingStore
from knowledge import Chunk
from search import SearchResult


TOKEN_LIMIT = 8000  # Запас до окна модели 8192.
QUERY_INSTRUCTION = "Instruct: Given a query, retrieve relevant passages\nQuery: "


@cache
def load_tokenizer() -> Tokenizer:
    """Загрузить только токенизатор фиксированной версии, без весов модели."""
    tokenizer = Tokenizer.from_pretrained(
        "ai-sage/Giga-Embeddings-instruct-3B-0826",
        revision="b71168088212f0a13688514e4fc288a86106c4e4",
    )
    tokenizer.no_truncation()
    tokenizer.no_padding()
    return tokenizer


def split_for_embedding(text: str) -> list[str]:
    """Разбить текст без потерь; каждая часть укладывается в лимит токенов."""
    tokenizer = load_tokenizer()
    if text and len(tokenizer.encode(text).ids) <= TOKEN_LIMIT:
        return [text]
    parts = []
    start = 0
    while start < len(text):
        low, high = 1, len(text) - start
        best = 0
        while low <= high:
            size = (low + high) // 2
            count = len(tokenizer.encode(text[start:start + size]).ids)
            if count <= TOKEN_LIMIT:
                best = size
                low = size + 1
            else:
                high = size - 1
        if best == 0:
            raise ValueError("Не удалось вместить символ в окно эмбеддингов")
        parts.append(text[start:start + best])
        start += best
    return parts


def normalize(vectors: np.ndarray) -> np.ndarray:
    """Нормировать строки матрицы для вычисления косинусного сходства."""
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("Ожидается конечная двумерная матрица")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Нулевой вектор нельзя нормировать")
    return matrix / norms


class SemanticSearch:
    """Один нормированный вектор на фрагмент; индекс находится в памяти."""

    def __init__(self, chunks: list[Chunk], vectors: np.ndarray, store: EmbeddingStore) -> None:
        """Связать готовую матрицу с фрагментами и клиентом запросов."""
        if not chunks or len(chunks) != len(vectors):
            raise ValueError("Нужен один вектор на каждый фрагмент непустой базы")
        if len({chunk.chunk_id for chunk in chunks}) != len(chunks):
            raise ValueError("Идентификаторы фрагментов должны быть уникальны")
        self.chunks = tuple(chunks)
        self.vectors = normalize(vectors)
        self.store = store

    @classmethod
    def build(cls, chunks: list[Chunk], store: EmbeddingStore) -> "SemanticSearch":
        """Проиндексировать все фрагменты с использованием локального кэша.

        Длинный текст режется по лимиту токенов модели без потери текста.
        Нормированные векторы частей усредняются с весами по числу символов.
        Это простая агрегация: она может размывать тему длинного фрагмента.
        """
        vectors = []
        for offset in range(0, len(chunks), 64):
            groups = []
            for chunk in chunks[offset:offset + 64]:
                parts = split_for_embedding(chunk.text)
                if not parts:
                    raise ValueError(f"Пустой текст фрагмента: {chunk.chunk_id}")
                groups.append(parts)
            matrix = normalize(store.embed([part for parts in groups for part in parts], input_type="passage"))
            start = 0
            for parts in groups:
                end = start + len(parts)
                vectors.append(np.average(matrix[start:end], axis=0, weights=[len(part) for part in parts]))
                start = end
        return cls(chunks, np.asarray(vectors), store)

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Получить эмбеддинг вопроса и ранжировать фрагменты по косинусу.

        Непустой вопрос может вернуть результаты даже без фактического ответа.
        Порог релевантности не выбран; score не является уверенностью модели.
        """
        if top_k < 1:
            raise ValueError("top_k должен быть положительным")
        if not query.strip():
            return []
        instructed_query = QUERY_INSTRUCTION + query
        if len(load_tokenizer().encode(instructed_query).ids) > TOKEN_LIMIT:
            raise ValueError(f"Вопрос с инструкцией должен укладываться в {TOKEN_LIMIT} токенов")
        query_vector = normalize(self.store.embed([instructed_query], input_type="query"))[0]
        if query_vector.shape[0] != self.vectors.shape[1]:
            raise ValueError("Размерности эмбеддингов базы и вопроса не совпадают")
        scores = self.vectors @ query_vector
        order = np.argsort(-scores, kind="stable")[:top_k]
        return [SearchResult(self.chunks[index], float(scores[index])) for index in order]
