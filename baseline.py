"""Воспроизводимый прогон baseline на золотом наборе с полными трассами и usage."""

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from assistant import Answer, AnswerDraft, BaselineAssistant, BaselineConfig, system_prompt, token_totals, tool_definitions
from embeddings import EmbeddingStore, create_client
from evaluation import corpus_fingerprint, load_golden, reference_chunk_ids
from knowledge import Chunk
from search import LexicalSearch
from semantic import SemanticSearch


def _summary(rows: list[dict], expected_questions: int) -> dict:
    """Считать техническую валидность, не называть её фактической точностью."""
    valid = [row for row in rows if row["status"] == "ok"]
    chat_calls = [call for row in rows for call in row["chat_calls"]]
    embedding_calls = [call for row in rows for call in row["embedding_calls"]]
    usage = token_totals(chat_calls)
    return {
        "expected_questions": expected_questions, "attempted_questions": len(rows),
        "valid_responses": len(valid), "valid_response_rate": len(valid) / expected_questions,
        "answered_questions": sum(not row["answer"]["insufficient_data"] for row in valid),
        "insufficient_data_questions": sum(row["answer"]["insufficient_data"] for row in valid),
        "errors": sum(row["status"] != "ok" for row in rows),
        "tool_errors": sum(call["error"] is not None for row in rows for call in row["tool_calls"]),
        "gold_evidence_seen_rate": sum(row["gold_evidence_seen"] for row in rows) / expected_questions,
        "gold_evidence_cited_rate": sum(row["gold_evidence_cited"] for row in rows) / expected_questions,
        "mean_seconds": sum(row["seconds"] for row in rows) / len(rows) if rows else None,
        "chat_api_attempts": len(chat_calls), "chat_token_totals": usage,
        "query_embedding_api_attempts": len(embedding_calls),
        "query_embedding_token_totals": token_totals(embedding_calls),
        "known_chat_tokens_per_valid_response": usage["known_total_tokens"] / len(valid) if valid else None,
        "quality_scope": "Schema, verbatim citations and golden evidence coordinates; not factual answer accuracy.",
    }


def _save(path: Path, report: dict) -> None:
    """Атомарно сохранять прогресс после каждого вопроса, не теряя ранние трассы."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_baseline(
    project_dir: Path, chunks: list[Chunk], lexical_search: LexicalSearch,
    config: BaselineConfig | None = None, *, name: str = "baseline",
) -> tuple[dict, Path]:
    """Прогнать все золотые вопросы; эталоны не передаются ассистенту.

    Каждая запись содержит ответ или явную ошибку, вызовы модели и инструментов,
    фактический usage и сравнение с координатами эталона. Генерация ответов живая,
    не кэшируется: повторный запуск создаёт новый отчёт и расходует квоту.
    """
    cases = load_golden(project_dir / "data/golden.csv")
    targets = reference_chunk_ids(cases, chunks)
    config = config or BaselineConfig()
    started = datetime.now(timezone.utc)
    if not name or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for character in name):
        raise ValueError("Имя прогона должно содержать только a-z, 0-9, _ и -")
    path = project_dir / "results" / f"{name}_{started:%Y%m%dT%H%M%S%fZ}.json"
    report = {
        "status": "running", "cases": [],
        "metadata": {
            "started_at_utc": started.isoformat(), "name": name, "config": asdict(config),
            "questions": len(cases), "chunks": len(chunks),
            "system_prompt": system_prompt(config), "tools": tool_definitions(),
            "answer_schema": Answer.model_json_schema(),
            "model_answer_schema": AnswerDraft.model_json_schema(),
            "golden_sha256": hashlib.sha256((project_dir / "data/golden.csv").read_bytes()).hexdigest(),
            "corpus_sha256": corpus_fingerprint(chunks),
            "implementation_sha256": {
                name: hashlib.sha256((project_dir / name).read_bytes()).hexdigest()
                for name in ("assistant.py", "context.py", "baseline.py", "embeddings.py", "semantic.py", "search.py", "evaluation.py", "knowledge.py")
            },
            "chat_retry_policy": "No automatic retries or JSON repair; at most 2 tool-selection calls and 1 final call.",
            "embedding_retry_policy": "EmbeddingStore defaults; each actual attempt is recorded separately.",
            "evaluation_note": "Gold answers and source coordinates are report labels, not model inputs.",
        },
    }
    _save(path, report)
    store = None
    try:
        with create_client(project_dir) as client:
            store = EmbeddingStore(project_dir / ".local/embeddings-giga-3b.sqlite", client)
            report["metadata"]["base_url"] = str(client.base_url)
            report["metadata"]["embedding_model"] = store.model
            before_index = len(store.usage())
            try:
                semantic_search = SemanticSearch.build(chunks, store)
            finally:
                report["metadata"]["index_embedding_calls"] = store.usage()[before_index:]
            assistant = BaselineAssistant(client, lexical_search, semantic_search, config)
            _save(path, report)
            for case in cases:
                before = len(store.usage())
                row = assistant.answer(case.question)
                row["embedding_calls"] = store.usage()[before:]
                cited = {citation["chunk_id"] for citation in row["answer"]["citations"]} if row["answer"] else set()
                row.update({
                    "case_id": case.case_id, "expected_answer": case.answer,
                    "expected_source": case.source_id, "expected_timecode": case.timecode,
                    "reference_chunk_ids": sorted(targets[case.case_id]),
                    "gold_evidence_seen": bool(targets[case.case_id].intersection(row["retrieved_chunk_ids"])),
                    "gold_evidence_cited": bool(targets[case.case_id].intersection(cited)),
                })
                report["cases"].append(row)
                report["summary"] = _summary(report["cases"], len(cases))
                _save(path, report)
                print(f"{name}: вопрос {case.case_id}/{len(cases)} — {row['status']}", flush=True)
            report["status"] = "complete"
    except Exception as exc:
        report["status"] = "error"
        report["error"] = type(exc).__name__
    finally:
        report["metadata"]["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        report["summary"] = _summary(report["cases"], len(cases))
        _save(path, report)
        if store is not None:
            store.close()
    return report, path
