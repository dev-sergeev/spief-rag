"""Дословные окна контекста и отбор по исходному вопросу без эталонных данных."""

from typing import Literal

from search import tokenize


ContextMode = Literal["full", "deduplicated", "focused", "compact"]

# Служебные слова вопроса не должны вытеснять имена, числа и предмет ответа.
_QUESTION_WORDS = frozenset(
    "а в во для до и из или к как какие какой какая какое каких кем кто ли на "
    "о об от по почему при про с со у что чем чего чьи это этот эта эти".split()
)


def excerpts(text: str) -> list[str]:
    """Покрыть текст исходными окнами до 900 символов с перекрытием 120.

    Границы и strip совпадают с замороженным baseline. Каждая выдержка остаётся
    непрерывной дословной подстрокой, без склейки предложений или многоточий.
    """
    windows = []
    start = 0
    while start < len(text):
        end = min(start + 900, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start + 450, end)
            if boundary > start:
                end = boundary
        windows.append(text[start:end].strip())
        if end == len(text):
            break
        start = end - 120
    return windows


def select_excerpts(text: str, question: str, mode: ContextMode) -> list[str]:
    """Выбрать все окна либо не более двух релевантных, сохранив их порядок.

    Сначала выбирается окно с наибольшим числом различных содержательных токенов
    вопроса, затем — окно с наибольшим числом ещё не покрытых токенов. При равенстве
    важнее полное число совпадений, затем исходная позиция. Повтор слова не повышает
    оценку. Короткие фрагменты не обрезаются. Без совпадений остаются первые два
    окна, чтобы не терять смысловую выдачу. Имена файлов, координаты, запросы
    модели и golden-ответы сюда не поступают.
    """
    if mode not in ("full", "deduplicated", "focused", "compact"):
        raise ValueError(f"Неизвестный режим контекста: {mode}")
    windows = excerpts(text)
    if mode != "focused" or len(windows) <= 2:
        return windows
    terms = set(tokenize(question)) - _QUESTION_WORDS
    matches = [terms.intersection(tokenize(window)) for window in windows]
    selected = []
    covered = set()
    for _ in range(2):
        index = min(
            (index for index in range(len(windows)) if index not in selected),
            key=lambda index: (-len(matches[index] - covered), -len(matches[index]), index),
        )
        selected.append(index)
        covered.update(matches[index])
    return [windows[index] for index in sorted(selected)]
