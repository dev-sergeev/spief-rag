"""Фактические токены замороженного baseline и качество без оценки стоимости."""

from assistant import token_totals


def summarize_calls(calls: list[dict]) -> dict:
    """Суммировать все попытки API, включая неудачные, в родных токенах модели.

    Токен — единица конкретного токенизатора, не универсальная мера текста.
    Поэтому разные явно указанные модели в одном списке запрещены. Чат,
    query embeddings, index embeddings и оценщик должны иметь отдельные списки.
    Кэшированные входные токены не вычитаются. known_* сохраняют известные
    частичные суммы; total_tokens известен только при usage.total_tokens у
    каждой попытки. Пустой список означает ноль попыток и ноль токенов.
    """
    models = {call["model"] for call in calls if call.get("model") is not None}
    if len(models) > 1:
        raise ValueError("Нельзя суммировать токены разных моделей в одном списке")
    summary = token_totals(calls)
    summary["api_attempts"] = len(calls)
    summary["total_tokens"] = (
        summary["known_total_tokens"]
        if summary["unknown_total_tokens_calls"] == 0 else None
    )
    return summary


def _rows_by_id(rows: list[dict], label: str) -> dict[int, dict]:
    """Построить однозначное соответствие строк вопросам без потери знаменателя."""
    indexed = {}
    for row in rows:
        case_id = row["case_id"]
        if type(case_id) is not int:
            raise ValueError(f"{label}: case_id должен быть целым числом")
        if case_id in indexed:
            raise ValueError(f"{label}: повторяющийся case_id {case_id}")
        indexed[case_id] = row
    return indexed


def summarize_efficiency(baseline: dict, assessments: list[dict]) -> dict:
    """Связать оценки всех вопросов с расходом только исходного baseline.

    content_correct считает фактически верные ответы, supported_correct —
    success (верность вместе с локальной валидностью и подтверждением).
    Ошибки pipeline и отказы — известные неуспехи на полностью ответном golden;
    judge_error и unjudged — неизвестный исход, а не ошибка ассистента.
    unjudged учитывается отдельно: это ещё не попытка оценки. Точная доля при
    неизвестных метках равна None. Нижняя граница считает их неуспехами,
    верхняя — успехами; знаменатель всегда включает все вопросы baseline.

    Числитель отношений включает все chat_calls, в том числе неверные ответы
    и неудачные API-попытки. Неизвестный расход, неизвестная соответствующая
    метка или ноль успехов дают None, а не искусственный ноль. При нуле
    вопросов доли и отношения также None. Эмбеддинги запроса и индекса
    считаются отдельно в родных токенах своих моделей; расходы оценщика и
    сохранённые summary baseline не используются. Входные данные не меняются.
    """
    cases = baseline["cases"]
    cases_by_id = _rows_by_id(cases, "baseline")
    assessments_by_id = _rows_by_id(assessments, "assessments")
    if cases_by_id.keys() != assessments_by_id.keys():
        missing = sorted(cases_by_id.keys() - assessments_by_id.keys())
        extra = sorted(assessments_by_id.keys() - cases_by_id.keys())
        raise ValueError(f"Несовпадение case_id: missing={missing}, extra={extra}")

    questions = len(cases)
    quality = {
        "content_correct": 0,
        "supported_correct": 0,
        "content_unknown": 0,
        "supported_unknown": 0,
        "pipeline_errors": 0,
        "abstentions": 0,
        "unjudged": 0,
        "judge_errors": 0,
    }
    status_counters = {
        "pipeline_error": "pipeline_errors",
        "abstained": "abstentions",
        "unjudged": "unjudged",
        "judge_error": "judge_errors",
    }
    for row in assessments_by_id.values():
        status = row["status"]
        if status == "graded":
            content = row["content_correct"]
            supported = row["success"]
            if any(value is not None and type(value) is not bool for value in (content, supported)):
                raise ValueError("Оценки content_correct и success должны быть bool или None")
            if supported is True and (content is not True or row.get("grounded") is not True):
                raise ValueError("success=True требует content_correct=True и grounded=True")
        elif status in ("pipeline_error", "abstained"):
            content = supported = False
        elif status in ("judge_error", "unjudged"):
            content = supported = None
        else:
            raise ValueError(f"Неизвестный статус оценки: {status}")
        if status in status_counters:
            quality[status_counters[status]] += 1
        quality["content_correct"] += content is True
        quality["supported_correct"] += supported is True
        quality["content_unknown"] += content is None
        quality["supported_unknown"] += supported is None

    for name, unknown_name in (
        ("content_correct", "content_unknown"),
        ("supported_correct", "supported_unknown"),
    ):
        correct, unknown = quality[name], quality[unknown_name]
        quality[f"{name}_rate"] = correct / questions if questions and not unknown else None
        quality[f"{name}_rate_lower_bound"] = correct / questions if questions else None
        quality[f"{name}_rate_upper_bound"] = (
            (correct + unknown) / questions if questions else None
        )

    assistant_chat = summarize_calls([call for case in cases for call in case.get("chat_calls", [])])
    total = assistant_chat["total_tokens"]
    assistant_chat["tokens_per_question"] = (
        total / questions if total is not None and questions else None
    )
    for key, correct_key, unknown_key in (
        ("tokens_per_correct_answer", "content_correct", "content_unknown"),
        ("tokens_per_supported_correct_answer", "supported_correct", "supported_unknown"),
    ):
        correct = quality[correct_key]
        assistant_chat[key] = (
            total / correct
            if total is not None and correct and not quality[unknown_key] else None
        )
    return {
        "questions": questions,
        "assistant_chat": assistant_chat,
        "query_embeddings": summarize_calls([
            call for case in cases for call in case.get("embedding_calls", [])
        ]),
        "index_embeddings": summarize_calls(baseline.get("metadata", {}).get("index_embedding_calls", [])),
        "quality": quality,
    }
