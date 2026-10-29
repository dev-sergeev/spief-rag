"""Загрузка стенограмм ПМЭФ и разбиение по исходным таймкодам."""

from dataclasses import dataclass
from pathlib import Path
import re


SEGMENT_PATTERN = re.compile(
    r"^\[(\d{2}:\d{2}:\d{2}) - (\d{2}:\d{2}:\d{2})\] \| ([^|]+) \| (.+)$"
)


@dataclass(frozen=True)
class Document:
    """Исходная стенограмма; source_id совпадает с именем файла."""

    source_id: str
    text: str


@dataclass(frozen=True)
class Chunk:
    """Фрагмент с исходным текстом и координатами для ссылки на источник."""

    chunk_id: str
    source_id: str
    start_time: str
    end_time: str
    speaker: str
    text: str


def load_documents(directory: Path) -> list[Document]:
    """Прочитать TXT-файлы в стабильном порядке, не изменяя исходники."""
    if not directory.is_dir():
        raise FileNotFoundError(f"Нет папки со стенограммами: {directory}")
    paths = sorted(directory.glob("*.txt"))
    if not paths:
        raise ValueError(f"В папке нет TXT-файлов: {directory}")
    documents = []
    for path in paths:
        text = path.read_text(encoding="utf-8-sig")
        if not text.strip():
            raise ValueError(f"Пустая стенограмма: {path.name}")
        documents.append(Document(source_id=path.name, text=text))
    return documents


def split_documents(documents: list[Document]) -> list[Chunk]:
    """Сделать каждую непустую строку с таймкодом отдельным фрагментом.

    Строка сохраняется целиком, включая таймкод и имя выступающего.
    Неизвестный формат вызывает ошибку вместо молчаливой потери текста.
    Идентификаторы воспроизводимы при неизменных файлах и их именах.
    """
    chunks = []
    for document in documents:
        segment_number = 0
        for line_number, line in enumerate(document.text.splitlines(), start=1):
            if not line.strip():
                continue
            match = SEGMENT_PATTERN.fullmatch(line)
            if match is None:
                raise ValueError(
                    f"Неизвестный формат: {document.source_id}, строка {line_number}"
                )
            segment_number += 1
            start_time, end_time, speaker, _ = match.groups()
            chunks.append(Chunk(
                chunk_id=f"{document.source_id}#{segment_number:04d}",
                source_id=document.source_id,
                start_time=start_time,
                end_time=end_time,
                speaker=speaker.strip(),
                text=line,
            ))
    return chunks
