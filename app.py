"""Финальный запуск того же RAG-пайплайна на произвольных вопросах из ноутбука."""

from collections.abc import Sequence
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from assistant import BaselineConfig, token_totals
from embeddings import EmbeddingStore, create_client
from evaluation import corpus_fingerprint
from experiments import ExecutionConfig, create_assistant, strategy_prompt
from knowledge import load_documents, split_documents
from search import LexicalSearch
from semantic import SemanticSearch


def _save(path: Path, report: dict) -> None:
    """Сохранить полную трассу атомарно, не публикуя частичный JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run_demo(
    project_dir: Path, config: BaselineConfig, questions: Sequence[str],
    *, execution: ExecutionConfig | None = None,
) -> tuple[dict, Path]:
    """Ответить на вопросы с одной загрузкой индекса и отдельным состоянием ответа.

    Конфигурацию и способ исполнения выбирает проверенное сравнение экспериментов;
    без execution используется исходный baseline. Генерация всегда живая:
    кэшируются только эмбеддинги. Каждый вопрос, ошибка и фактический usage
    сохраняются. Это демонстрация работы, не новый замер качества: эталоны
    и рубрика в генерацию не передаются.
    """
    project_dir = Path(project_dir)
    if isinstance(questions, str) or not questions:
        raise ValueError("Передайте непустой список вопросов, а не одну строку")
    if any(not isinstance(question, str) or not question.strip() for question in questions):
        raise ValueError("Каждый вопрос должен быть непустой строкой")
    questions = list(questions)
    execution = execution if execution is not None else ExecutionConfig()
    chunks = split_documents(load_documents(project_dir / "data/SPIEF_txt"))
    lexical = LexicalSearch(chunks)
    started = datetime.now(timezone.utc)
    path = project_dir / "results" / f"demo_selected_{started:%Y%m%dT%H%M%S%fZ}.json"
    report = {
        "status": "running", "error": None, "cases": [],
        "metadata": {
            "started_at_utc": started.isoformat(), "config": asdict(config),
            "execution": asdict(execution), "system_prompt": strategy_prompt(config, execution),
            "corpus_sha256": corpus_fingerprint(chunks),
            "question_count": len(questions),
            "implementation_sha256": {
                name: hashlib.sha256((project_dir / name).read_bytes()).hexdigest()
                for name in ("app.py", "experiments.py", "assistant.py", "context.py", "embeddings.py", "semantic.py", "search.py", "knowledge.py")
            },
            "scope": "Live demonstration; no golden labels are supplied to the assistant.",
            "index_embedding_calls": [],
        },
    }
    _save(path, report)
    try:
        with create_client(project_dir) as client:
            store = EmbeddingStore(project_dir / ".local/embeddings-giga-3b.sqlite", client)
            try:
                report["metadata"]["embedding_model"] = store.model
                before_index = len(store.usage())
                try:
                    semantic = SemanticSearch.build(chunks, store)
                finally:
                    report["metadata"]["index_embedding_calls"] = store.usage()[before_index:]
                    _save(path, report)
                assistant = create_assistant(client, lexical, semantic, config, execution)
                for number, question in enumerate(questions, start=1):
                    before_question = len(store.usage())
                    row = assistant.answer(question)
                    row["scenario_id"] = number
                    row["embedding_calls"] = store.usage()[before_question:]
                    report["cases"].append(row)
                    _save(path, report)
                report["status"] = (
                    "complete" if all(row["status"] == "ok" for row in report["cases"])
                    else "complete_with_errors"
                )
            finally:
                store.close()
    except Exception as exc:
        report["status"] = "error"
        # Не сериализуем HTTP-запрос, заголовки или тело ошибки сервиса.
        report["error"] = type(exc).__name__
    finally:
        calls = [call for row in report["cases"] for call in row["chat_calls"]]
        query_calls = [call for row in report["cases"] for call in row["embedding_calls"]]
        report["summary"] = {
            "expected_questions": len(questions), "attempted_questions": len(report["cases"]),
            "pipeline_errors": sum(row["status"] != "ok" for row in report["cases"]),
            "chat_api_attempts": len(calls), "chat_token_totals": token_totals(calls),
            "query_embedding_token_totals": token_totals(query_calls),
        }
        report["metadata"]["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        _save(path, report)
    return report, path
