"""Оценка поиска по золотому набору: источник, таймкод и ранг фрагмента."""

from collections import defaultdict
from collections.abc import Callable
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time

from knowledge import Chunk
from search import LexicalSearch, SearchResult


@dataclass(frozen=True)
class GoldenCase:
    """Вопрос с эталонным ответом и координатами подтверждающего фрагмента."""

    case_id: int
    question: str
    answer: str
    source_id: str
    timecode: str
    quote: str
    category: str


def _seconds(timecode: str) -> int:
    """Преобразовать таймкод стенограммы в секунды; часы могут превышать 23."""
    if not re.fullmatch(r"\d{2,}:[0-5]\d:[0-5]\d", timecode):
        raise ValueError(f"Некорректный таймкод: {timecode}")
    hours, minutes, seconds = map(int, timecode.split(":"))
    return hours * 3600 + minutes * 60 + seconds


def load_golden(path: Path) -> list[GoldenCase]:
    """Прочитать исходный CSV, найдя заголовок после вводных строк таблицы."""
    columns = {"№", "Вопрос", "Эталонный ответ", "Файл-источник", "Таймкод", "Цитата-подтверждение", "Тип"}
    cases = []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        for header in reader:
            if columns.issubset(header):
                break
        else:
            raise ValueError("Не найден заголовок золотого набора")
        for values in reader:
            if not any(value.strip() for value in values):
                continue
            if len(values) != len(header):
                raise ValueError("Число столбцов не совпадает с заголовком золотого набора")
            row = dict(zip(header, (value.strip() for value in values)))
            if any(not row[column] for column in columns):
                raise ValueError(f"Незаполненное поле в вопросе {row['№']}")
            _seconds(row["Таймкод"])
            cases.append(GoldenCase(
                case_id=int(row["№"]), question=row["Вопрос"], answer=row["Эталонный ответ"],
                source_id=row["Файл-источник"], timecode=row["Таймкод"],
                quote=row["Цитата-подтверждение"], category=row["Тип"],
            ))
    if not cases or len({case.case_id for case in cases}) != len(cases):
        raise ValueError("Золотой набор пуст или содержит повторяющиеся номера")
    return cases


def reference_chunk_ids(cases: list[GoldenCase], chunks: list[Chunk]) -> dict[int, set[str]]:
    """Проверить наличие эталонов до вызовов API; интервалы имеют вид [start, end)."""
    by_source = defaultdict(list)
    for chunk in chunks:
        by_source[chunk.source_id].append(chunk)
    targets = {}
    for case in cases:
        moment = _seconds(case.timecode)
        matched = {
            chunk.chunk_id for chunk in by_source[case.source_id]
            if _seconds(chunk.start_time) <= moment < _seconds(chunk.end_time)
        }
        if not matched:
            raise ValueError(
                f"Нет эталонного фрагмента для вопроса {case.case_id}: "
                f"{case.source_id}, {case.timecode}"
            )
        targets[case.case_id] = matched
    return targets


def evaluate_retrieval(
    cases: list[GoldenCase], chunks: list[Chunk],
    search: Callable[[str, int], list[SearchResult]], method: str, top_k: int = 5,
) -> list[dict]:
    """Оценить каждый вопрос без передачи эталонных координат поисковику.

    Ошибка запроса записывается отдельной строкой с нулевым попаданием,
    а не исключается из знаменателя метрик. Ответы LLM здесь не оцениваются.
    """
    if top_k < 1 or not cases:
        raise ValueError("Нужны непустой набор вопросов и положительный top_k")
    targets = reference_chunk_ids(cases, chunks)
    rows = []
    for case in cases:
        started = time.perf_counter()
        error = None
        try:
            found = search(case.question, top_k)[:top_k]
        except Exception as exc:
            found = []
            error = f"{type(exc).__name__}: {exc}"
        elapsed = time.perf_counter() - started
        target_rank = next(
            (rank for rank, result in enumerate(found, 1)
             if result.chunk.chunk_id in targets[case.case_id]), None,
        )
        source_rank = next(
            (rank for rank, result in enumerate(found, 1)
             if result.chunk.source_id == case.source_id), None,
        )
        rows.append({
            "case_id": case.case_id, "question": case.question, "category": case.category,
            "method": method, "top_k": top_k,
            "expected_source": case.source_id, "expected_timecode": case.timecode,
            "expected_chunk_ids": sorted(targets[case.case_id]),
            "target_rank": target_rank, "source_rank": source_rank,
            "elapsed_seconds": elapsed, "error": error,
            "retrieved": [
                {"chunk_id": result.chunk.chunk_id, "source_id": result.chunk.source_id,
                 "start_time": result.chunk.start_time, "end_time": result.chunk.end_time,
                 "score": float(result.score)}
                for result in found
            ],
        })
    return rows


def summarize_retrieval(rows: list[dict]) -> list[dict]:
    """Посчитать Hit@1, Hit@k, MRR@k и попадание в файл на всех вопросах метода."""
    if not rows:
        raise ValueError("Нет результатов оценки")
    groups = defaultdict(list)
    for row in rows:
        groups[row["method"]].append(row)
    summary = []
    for method, group in groups.items():
        cutoffs = {row["top_k"] for row in group}
        if len(cutoffs) != 1 or len({row["case_id"] for row in group}) != len(group):
            raise ValueError("Метрики метода требуют одного top_k и уникальных вопросов")
        count = len(group)
        summary.append({
            "method": method, "questions": count, "top_k": group[0]["top_k"],
            "hit_at_1": sum(row["target_rank"] == 1 for row in group) / count,
            "hit_at_k": sum(row["target_rank"] is not None for row in group) / count,
            "mrr_at_k": sum(1 / row["target_rank"] if row["target_rank"] else 0 for row in group) / count,
            "source_hit_at_k": sum(row["source_rank"] is not None for row in group) / count,
            "errors": sum(row["error"] is not None for row in group),
            "mean_seconds": sum(row["elapsed_seconds"] for row in group) / count,
        })
    return summary


def corpus_fingerprint(chunks: list[Chunk]) -> str:
    """Получить SHA-256 текстов и идентификаторов в фактическом порядке индекса."""
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(json.dumps([chunk.chunk_id, chunk.text], ensure_ascii=False).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def save_retrieval_report(path: Path, rows: list[dict], summary: list[dict], metadata: dict) -> None:
    """Сохранить метрики, ранжирование каждого вопроса и настройки запуска в JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {"metadata": metadata, "summary": summary, "cases": rows}
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run_retrieval_comparison(
    project_dir: Path, chunks: list[Chunk], lexical_search: LexicalSearch, top_k: int = 5,
) -> tuple[dict, Path]:
    """Проверить оба поиска, закрыть API-ресурсы и сохранить отдельный отчёт запуска.

    Индекс восстанавливается из кэша. Новые эмбеддинги вопросов сохраняются
    там же; после первого замера повторное выполнение может быть полностью
    локальным. Это отражается в числе реальных API-попыток отчёта.
    """
    from embeddings import EmbeddingStore, create_client
    from semantic import QUERY_INSTRUCTION, SemanticSearch

    golden_path = project_dir / "data/golden.csv"
    cases = load_golden(golden_path)
    reference_chunk_ids(cases, chunks)
    started_at = datetime.now(timezone.utc)
    client = create_client(project_dir)
    store = EmbeddingStore(project_dir / ".local/embeddings-giga-3b.sqlite", client)
    try:
        before = len(store.usage())
        semantic_search = SemanticSearch.build(chunks, store)
        after_index = len(store.usage())
        rows = evaluate_retrieval(cases, chunks, lexical_search.search, "BM25", top_k)
        rows += evaluate_retrieval(cases, chunks, semantic_search.search, "Giga Embeddings", top_k)
        calls = store.usage()
        metadata = {
            "started_at_utc": started_at.isoformat(),
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "embedding_model": store.model, "embedding_base_url": str(client.base_url),
            "query_instruction": QUERY_INSTRUCTION,
            "top_k": top_k, "questions": len(cases), "chunks": len(chunks),
            "golden_sha256": hashlib.sha256(golden_path.read_bytes()).hexdigest(),
            "corpus_sha256": corpus_fingerprint(chunks),
            "implementation_sha256": {
                name: hashlib.sha256((project_dir / name).read_bytes()).hexdigest()
                for name in ("knowledge.py", "search.py", "semantic.py", "embeddings.py", "evaluation.py")
            },
            "lexical_parameters": {"k1": 1.5, "b": 0.75, "stemming": False},
            "target_rule": "source_id matches and start_time <= golden timecode < end_time",
            "timing_note": "Per-question wall time; includes query embedding/cache, excludes index build.",
            "index_embedding_calls": calls[before:after_index],
            "query_embedding_calls": calls[after_index:],
        }
    finally:
        store.close()
        client.close()
    summary = summarize_retrieval(rows)
    stamp = started_at.strftime("%Y%m%dT%H%M%S%fZ")
    path = project_dir / "results" / f"retrieval_{stamp}.json"
    save_retrieval_report(path, rows, summary, metadata)
    return {"metadata": metadata, "summary": summary, "cases": rows}, path
